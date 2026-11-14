"""Home Assistant adapter for the Daikin Onecta API client."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta
import logging
from typing import Any

import aiohttp
from homeassistant import config_entries, core
from homeassistant.exceptions import OAuth2TokenRequestError, OAuth2TokenRequestReauthError
from homeassistant.helpers import config_entry_oauth2_flow, issue_registry as ir
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from daikin_onecta.client import OnectaClient
from daikin_onecta.exceptions import OnectaApiError, OnectaAuthenticationError, OnectaConnectionError, OnectaRateLimitError
from daikin_onecta.models import GatewayDevice
from daikin_onecta.rate_limit import RateLimit

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# POOL: how long to skip a pool member after a failure without a Retry-After value.
POOL_DEFAULT_BLOCK = timedelta(seconds=60)
POOL_TOKEN_ERROR_BLOCK = timedelta(minutes=5)
# POOL: how long to skip an account whose day counter shows 0 when Daikin
# gives no usable reset time. Daikin adds calls made while rate limited to the
# next window, so the code must not probe an exhausted account often.
POOL_DAY_EXHAUSTED_BLOCK = timedelta(hours=1)

# POOL: event fired once when no account can take calls any more.
EVENT_POOL_EXHAUSTED = f"{DOMAIN}_rate_limited"

BLOCK_STORE_KEY = f"{DOMAIN}.rate_limit_blocks"
BLOCK_STORE_VERSION = 1
BLOCK_STORE_DELAY = 5


async def async_load_blocks(hass: core.HomeAssistant) -> None:
    """Load saved account blocks, so that they survive a restart."""
    data = hass.data.setdefault(DOMAIN, {})
    if "block_store" in data:
        return
    store: Store[dict[str, str]] = Store(hass, BLOCK_STORE_VERSION, BLOCK_STORE_KEY)
    data["block_store"] = store
    data["blocks"] = await store.async_load() or {}


class DaikinApi:
    """Home Assistant adapter around the standalone Daikin Onecta client."""

    def __init__(
        self,
        hass: core.HomeAssistant,
        entry: config_entries.ConfigEntry,
        implementation: config_entry_oauth2_flow.AbstractOAuth2Implementation,
        pool: Mapping[str, "DaikinApi"] | None = None,
    ) -> None:
        """Initialize a new Daikin Onecta API.

        POOL: ``pool`` is a shared registry of DaikinApi objects of other
        config entries (other Daikin accounts) that lend their rate limit to
        this one. Members only supply a client; all locking and bookkeeping
        stays on this (primary) object.
        """
        self.hass = hass
        self._pool = pool if pool is not None else {}
        self._config_entry = entry
        self._blocked_until: datetime | None = None
        saved = hass.data.get(DOMAIN, {}).get("blocks", {}).get(entry.entry_id)
        if saved:
            self._blocked_until = dt_util.parse_datetime(saved)
        self._pool_exhausted = False
        self.session = config_entry_oauth2_flow.OAuth2Session(hass, entry, implementation)
        self._client = OnectaClient(async_get_clientsession(hass), self.async_get_access_token)

        # The Daikin cloud can return stale settings immediately after a PATCH.
        # The coordinator uses this timestamp to delay refreshes for its
        # configured scan-ignore period after a successful command.
        self._last_patch_call: datetime | None = None

        # The following lock is used to serialize http requests to Daikin cloud
        # to prevent receiving old settings while a PATCH is ongoing.
        self._cloud_lock = asyncio.Lock()

    @property
    def own_rate_limits(self) -> dict[str, int | None]:
        """Return the rate limits of this account only."""
        rate_limit = self._client.rate_limit
        return {
            "minute": rate_limit.minute_limit,
            "day": rate_limit.day_limit,
            "remaining_minutes": rate_limit.minute_remaining,
            "remaining_day": rate_limit.day_remaining,
            "retry_after": rate_limit.retry_after,
            "ratelimit_reset": rate_limit.reset,
        }

    @property
    def rate_limits(self) -> dict[str, int | None]:
        """Return rate limits summed over all accounts in the pool.

        POOL: limits and remaining counts are added; retry_after and reset
        give the soonest value. With no pool members this equals
        own_rate_limits.
        """
        members = self.pool_members()
        if len(members) == 1:
            return self.own_rate_limits
        all_limits = [member.own_rate_limits for member in members]
        result: dict[str, int | None] = {}
        for key in ("minute", "day", "remaining_minutes", "remaining_day"):
            values = [limits[key] for limits in all_limits if limits[key] is not None]
            result[key] = sum(values) if values else None
        for key in ("retry_after", "ratelimit_reset"):
            values = [limits[key] for limits in all_limits if limits[key] is not None]
            result[key] = min(values) if values else None
        return result

    @property
    def pool_status(self) -> list[dict[str, Any]]:
        """Return per-account state for diagnostics."""
        return [
            {
                "entry_id": member.config_entry.entry_id,
                "title": member.config_entry.title,
                "primary": member is self,
                "blocked_until": member.blocked_until.isoformat() if member.blocked_until else None,
                "rate_limits": member.own_rate_limits,
            }
            for member in self.pool_members()
        ]

    @property
    def config_entry(self) -> config_entries.ConfigEntry:
        """Return the config entry of this account."""
        return self._config_entry

    @property
    def blocked_until(self) -> datetime | None:
        """Return until when the pool skips this account."""
        return self._blocked_until

    def block(self, seconds: int | None, default: timedelta = POOL_DEFAULT_BLOCK) -> None:
        """Skip this account for a period. A longer existing block stays."""
        delta = timedelta(seconds=seconds) if seconds else default
        until = dt_util.utcnow() + delta
        if self._blocked_until is None or until > self._blocked_until:
            self._blocked_until = until
            self._save_block()

    def clear_block(self) -> None:
        """Stop skipping this account."""
        if self._blocked_until is not None:
            self._blocked_until = None
            self._save_block()

    def _save_block(self) -> None:
        """Write the block time of this account to storage."""
        data = self.hass.data.get(DOMAIN, {})
        store: Store | None = data.get("block_store")
        if store is None:
            return
        blocks: dict[str, str] = data.setdefault("blocks", {})
        if self._blocked_until is None:
            blocks.pop(self._config_entry.entry_id, None)
        else:
            blocks[self._config_entry.entry_id] = self._blocked_until.isoformat()
        store.async_delay_save(lambda: dict(blocks), BLOCK_STORE_DELAY)

    def block_if_exhausted(self) -> None:
        """Block this account when its last response shows no quota left.

        POOL: Daikin can accept a few calls after the counter shows 0, but
        it adds them to the next window. So the code stops at 0 and does not
        wait for the 429.
        """
        limits = self.own_rate_limits
        reset = limits["ratelimit_reset"]
        if limits["remaining_day"] == 0:
            # RateLimit-Reset can be the minute window. Only trust a value
            # that is longer than one minute for the day window.
            if reset is not None and reset > 60:
                self.block(reset)
            else:
                self.block(None, POOL_DAY_EXHAUSTED_BLOCK)
        elif limits["remaining_minutes"] == 0:
            self.block(reset if reset and reset <= 60 else None)

    def is_blocked(self) -> bool:
        """Return whether the pool skips this account now."""
        return self._blocked_until is not None and dt_util.utcnow() < self._blocked_until

    def pool_members(self) -> list["DaikinApi"]:
        """Return this account followed by all pool member accounts."""
        return [self, *(api for api in self._pool.values() if api is not self)]

    def _ordered_members(self) -> list["DaikinApi"]:
        """Return the accounts that may take a call, best first.

        POOL: blocked accounts are left out. Daikin adds calls made while
        rate limited to the next window, so a call to a blocked account
        makes the block longer. The account with the most remaining day
        quota comes first (unknown counts as most, so that a new member gets
        measured). The primary wins a tie, because sort is stable.
        """
        big = 1 << 30

        def key(member: "DaikinApi") -> tuple[int, int]:
            limits = member.own_rate_limits
            day = limits["remaining_day"]
            minute = limits["remaining_minutes"]
            return (
                -(big if day is None else day),
                -(big if minute is None else minute),
            )

        return sorted((m for m in self.pool_members() if not m.is_blocked()), key=key)

    def _pool_rate_error(self) -> OnectaRateLimitError:
        """Return a local rate-limit error for when every account is blocked."""
        now = dt_util.utcnow()
        ends = [m.blocked_until for m in self.pool_members() if m.blocked_until is not None]
        retry_after = max(1, int((min(ends) - now).total_seconds())) if ends else int(POOL_DEFAULT_BLOCK.total_seconds())
        limits = self.rate_limits
        rate_limit = RateLimit(
            minute_limit=limits["minute"],
            day_limit=limits["day"],
            minute_remaining=limits["remaining_minutes"],
            day_remaining=limits["remaining_day"],
            retry_after=retry_after,
            reset=retry_after,
        )
        return OnectaRateLimitError(rate_limit, method="POOL", path="(no call sent)")

    def _set_pool_exhausted(self, exhausted: bool, retry_after: int | None = None) -> None:
        """Fire an event when the pool becomes exhausted."""
        if exhausted and not self._pool_exhausted:
            until = dt_util.utcnow() + timedelta(seconds=retry_after or 0)
            _LOGGER.warning("All Daikin accounts are rate limited until %s, no calls are sent", until.isoformat())
            self.hass.bus.async_fire(
                EVENT_POOL_EXHAUSTED,
                {
                    "entry_id": self._config_entry.entry_id,
                    "accounts": len(self.pool_members()),
                    "retry_after": retry_after,
                    "until": until.isoformat(),
                },
            )
        self._pool_exhausted = exhausted

    async def _pooled(self, call: Callable[[OnectaClient], Awaitable[Any]]) -> Any:
        """Run one API call on the best account and fail over on a rate limit.

        POOL: the caller must hold self._cloud_lock. A rate-limit error or a
        token failure on one account moves the call to the next account.
        Errors of the primary account that are not about the rate limit are
        raised as before, so upstream behavior (reauth, UpdateFailed) stays.
        """
        members = self._ordered_members()
        if not members:
            err = self._pool_rate_error()
            self._set_pool_exhausted(True, err.retry_after)
            raise err
        last_rate_error: OnectaRateLimitError | None = None
        for member in members:
            is_primary = member is self
            try:
                result = await call(member.client)
            except OnectaRateLimitError as err:
                member.block(err.retry_after)
                last_rate_error = err
                if len(members) > 1:
                    _LOGGER.info(
                        "Daikin account '%s' is rate limited (retry after %s s), trying next account",
                        member.config_entry.title,
                        err.retry_after,
                    )
                continue
            except OnectaAuthenticationError:
                if is_primary:
                    raise
                _LOGGER.warning("Daikin pool account '%s' failed authentication, starting reauth", member.config_entry.title)
                member.block(None, POOL_TOKEN_ERROR_BLOCK)
                member.config_entry.async_start_reauth(self.hass)
                continue
            except OAuth2TokenRequestReauthError:
                if is_primary:
                    raise
                _LOGGER.warning("Daikin pool account '%s' needs reauthentication", member.config_entry.title)
                member.block(None, POOL_TOKEN_ERROR_BLOCK)
                member.config_entry.async_start_reauth(self.hass)
                continue
            except (OAuth2TokenRequestError, aiohttp.ClientError) as err:
                if is_primary:
                    raise
                _LOGGER.warning("Daikin pool account '%s' could not refresh its token: %s", member.config_entry.title, err)
                member.block(None, POOL_TOKEN_ERROR_BLOCK)
                continue
            member.clear_block()
            member.block_if_exhausted()
            self._set_pool_exhausted(False)
            if len(members) > 1:
                _LOGGER.debug("Daikin call served by account '%s'", member.config_entry.title)
            return result
        if last_rate_error is None:
            # Every account failed on its token. Report it as a rate limit,
            # so that the caller waits.
            err = self._pool_rate_error()
            self._set_pool_exhausted(True, err.retry_after)
            raise err
        if not self._ordered_members():
            self._set_pool_exhausted(True, self._pool_rate_error().retry_after)
        raise last_rate_error

    @property
    def client(self) -> OnectaClient:
        """Return the underlying Onecta client."""
        return self._client

    @property
    def last_patch_call(self) -> datetime | None:
        """Return when the last successful cloud write completed."""
        return self._last_patch_call

    async def async_get_access_token(self) -> str:
        """Return a valid OAuth access token."""
        await self.session.async_ensure_token_valid()
        return self.session.token["access_token"]

    def update_rate_limit_issues(self) -> None:
        """Update Home Assistant repair issues from the library rate-limit state."""
        limits = self.rate_limits
        if limits["remaining_minutes"] is not None and limits["remaining_minutes"] > 0:
            ir.async_delete_issue(self.hass, DOMAIN, "minute_rate_limit")
        if limits["remaining_day"] is not None and limits["remaining_day"] > 0:
            ir.async_delete_issue(self.hass, DOMAIN, "day_rate_limit")

    def create_rate_limit_issues(self) -> None:
        """Create Home Assistant repair issues for exhausted rate limits."""
        limits = self.rate_limits
        learn_more_url = (
            "https://developer.cloud.daikineurope.com/docs/b0dffcaa-7b51-428a-bdff-a7c8a64195c0/general_api_guidelines#doc-heading-rate-limitation"
        )
        if limits["remaining_minutes"] == 0:
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                "minute_rate_limit",
                is_fixable=False,
                is_persistent=True,
                severity=ir.IssueSeverity.ERROR,
                learn_more_url=learn_more_url,
                translation_key="minute_rate_limit",
            )
        if limits["remaining_day"] == 0:
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                "day_rate_limit",
                is_fixable=False,
                is_persistent=True,
                severity=ir.IssueSeverity.ERROR,
                learn_more_url=learn_more_url,
                translation_key="day_rate_limit",
            )

    async def get_cloud_device_details(self) -> list[GatewayDevice]:
        """Get typed device data from the Daikin cloud."""
        async with self._cloud_lock:
            try:
                devices = await self._pooled(lambda client: client.get_gateway_devices())
            except OnectaRateLimitError:
                self.create_rate_limit_issues()
                raise
            self.update_rate_limit_issues()
            return devices

    async def async_execute_command(self, command: Callable[[OnectaClient], Awaitable[None]]) -> bool:
        """Execute a serialized cloud command and handle expected failures."""
        async with self._cloud_lock:
            try:
                await self._pooled(command)
            except OnectaRateLimitError as err:
                self.create_rate_limit_issues()
                _LOGGER.warning(
                    "Daikin request %s %s was rate limited; retry after %s seconds",
                    err.method,
                    err.path,
                    err.retry_after,
                )
                return False
            except OnectaConnectionError as err:
                _LOGGER.warning(
                    "Daikin request %s %s failed to connect: %s",
                    err.method,
                    err.path,
                    err,
                )
                return False
            except OnectaApiError as err:
                _LOGGER.warning(
                    "Daikin request %s %s failed with HTTP %s",
                    err.method,
                    err.path,
                    err.status,
                )
                return False
            except TimeoutError:
                _LOGGER.warning("Daikin request timed out")
                return False
            self._last_patch_call = dt_util.utcnow()
            self.update_rate_limit_issues()
            return True
