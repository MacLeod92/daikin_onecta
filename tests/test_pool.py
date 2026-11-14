"""Tests for the multi-account API pool (local fork feature)."""

# ruff: noqa: SLF001

from unittest.mock import AsyncMock, MagicMock

from daikin_onecta.exceptions import OnectaAuthenticationError, OnectaRateLimitError
from daikin_onecta.rate_limit import RateLimit
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
import pytest
from pytest_homeassistant.common import MockConfigEntry

from custom_components.daikin_onecta import _remove_pool_member_devices
from custom_components.daikin_onecta.const import DOMAIN
from custom_components.daikin_onecta.daikin_api import EVENT_POOL_EXHAUSTED, DaikinApi, async_load_blocks


def _api(hass: HomeAssistant, entry_id: str, day: int | None, pool: dict | None = None) -> DaikinApi:
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.title = entry_id
    entry.data = {"token": {"access_token": "x"}}
    api = DaikinApi(hass, entry, MagicMock(), pool)
    client = MagicMock()
    client.rate_limit = RateLimit(minute_limit=20, day_limit=200, minute_remaining=20, day_remaining=day)
    client.get_gateway_devices = AsyncMock(return_value=[entry_id])
    api._client = client
    return api


def _rate_error(retry_after: int = 30) -> OnectaRateLimitError:
    return OnectaRateLimitError(RateLimit(minute_remaining=0, day_remaining=0, retry_after=retry_after), method="GET", path="/x")


async def test_no_pool_uses_own_client(hass: HomeAssistant) -> None:
    """Without members the API behaves like upstream."""
    primary = _api(hass, "primary", 150)
    assert await primary.get_cloud_device_details() == ["primary"]
    assert primary.rate_limits == primary.own_rate_limits


async def test_pool_picks_most_remaining(hass: HomeAssistant) -> None:
    """The account with the most day quota left serves the call."""
    pool: dict = {}
    primary = _api(hass, "primary", 50, pool)
    pool["member"] = _api(hass, "member", 120)
    assert await primary.get_cloud_device_details() == ["member"]

    pool["member"]._client.rate_limit = RateLimit(day_remaining=10)
    assert await primary.get_cloud_device_details() == ["primary"]


async def test_pool_unknown_quota_is_tried_first(hass: HomeAssistant) -> None:
    """A member without rate-limit headers yet gets measured first."""
    pool: dict = {}
    primary = _api(hass, "primary", 150, pool)
    pool["member"] = _api(hass, "member", None)
    assert await primary.get_cloud_device_details() == ["member"]


async def test_pool_fails_over_on_rate_limit(hass: HomeAssistant) -> None:
    """A 429 on one account moves the call to the next and blocks the first."""
    pool: dict = {}
    primary = _api(hass, "primary", 100, pool)
    member = _api(hass, "member", 150)
    pool["member"] = member
    member._client.get_gateway_devices = AsyncMock(side_effect=_rate_error(60))

    assert await primary.get_cloud_device_details() == ["primary"]
    assert member.is_blocked()

    # While blocked, the member is skipped although it shows more quota.
    member._client.get_gateway_devices = AsyncMock(return_value=["member"])
    assert await primary.get_cloud_device_details() == ["primary"]


async def test_pool_all_rate_limited_raises(hass: HomeAssistant) -> None:
    """When every account is rate limited the error reaches the caller."""
    pool: dict = {}
    primary = _api(hass, "primary", 0, pool)
    pool["member"] = _api(hass, "member", 0)
    for api in (primary, pool["member"]):
        api._client.get_gateway_devices = AsyncMock(side_effect=_rate_error())
    with pytest.raises(OnectaRateLimitError):
        await primary.get_cloud_device_details()


async def test_pool_member_auth_error_starts_reauth(hass: HomeAssistant) -> None:
    """An auth failure on a member starts its reauth and uses the primary."""
    pool: dict = {}
    primary = _api(hass, "primary", 100, pool)
    member = _api(hass, "member", 150)
    pool["member"] = member
    member._client.get_gateway_devices = AsyncMock(side_effect=OnectaAuthenticationError(401, method="GET", path="/x"))

    assert await primary.get_cloud_device_details() == ["primary"]
    member.config_entry.async_start_reauth.assert_called_once_with(hass)


async def test_pool_primary_auth_error_raises(hass: HomeAssistant) -> None:
    """An auth failure on the primary is raised as upstream does."""
    pool: dict = {}
    primary = _api(hass, "primary", 150, pool)
    pool["member"] = _api(hass, "member", 100)
    primary._client.get_gateway_devices = AsyncMock(side_effect=OnectaAuthenticationError(401, method="GET", path="/x"))
    with pytest.raises(OnectaAuthenticationError):
        await primary.get_cloud_device_details()


async def test_pool_command_fails_over(hass: HomeAssistant) -> None:
    """A command retries on the next account after a 429."""
    pool: dict = {}
    primary = _api(hass, "primary", 100, pool)
    member = _api(hass, "member", 150)
    pool["member"] = member
    used = []

    async def command(client) -> None:
        if client is member.client:
            raise _rate_error()
        used.append(client)

    assert await primary.async_execute_command(command) is True
    assert used == [primary.client]
    assert primary.last_patch_call is not None


async def test_pool_rate_limits_are_summed(hass: HomeAssistant) -> None:
    """The rate-limit sensors show the pool total."""
    pool: dict = {}
    primary = _api(hass, "primary", 100, pool)
    pool["member"] = _api(hass, "member", 150)
    limits = primary.rate_limits
    assert limits["day"] == 400
    assert limits["remaining_day"] == 250
    assert limits["minute"] == 40
    assert len(primary.pool_status) == 2


async def test_pool_never_calls_blocked_accounts(hass: HomeAssistant) -> None:
    """When every account is blocked, no call goes out and one event fires."""
    pool: dict = {}
    primary = _api(hass, "primary", 100, pool)
    member = _api(hass, "member", 150)
    pool["member"] = member
    for api in (primary, member):
        api._client.get_gateway_devices = AsyncMock(side_effect=_rate_error(600))
    events = []
    hass.bus.async_listen(EVENT_POOL_EXHAUSTED, events.append)

    with pytest.raises(OnectaRateLimitError):
        await primary.get_cloud_device_details()
    assert primary._client.get_gateway_devices.await_count == 1
    assert member._client.get_gateway_devices.await_count == 1

    # Later calls and commands do not reach Daikin while blocked.
    with pytest.raises(OnectaRateLimitError) as err:
        await primary.get_cloud_device_details()
    assert 590 < err.value.retry_after <= 600
    command = AsyncMock()
    assert await primary.async_execute_command(command) is False
    command.assert_not_awaited()
    assert primary._client.get_gateway_devices.await_count == 1
    assert member._client.get_gateway_devices.await_count == 1

    await hass.async_block_till_done()
    assert len(events) == 1
    assert events[0].data["accounts"] == 2


async def test_pool_stops_at_zero_day_quota(hass: HomeAssistant) -> None:
    """An account whose response shows 0 day quota is not used again."""
    pool: dict = {}
    primary = _api(hass, "primary", 100, pool)
    member = _api(hass, "member", 150)
    pool["member"] = member

    async def last_call() -> list[str]:
        member._client.rate_limit = RateLimit(day_remaining=0, minute_remaining=10, reset=3600)
        return ["member"]

    member._client.get_gateway_devices = AsyncMock(side_effect=last_call)
    assert await primary.get_cloud_device_details() == ["member"]
    assert member.is_blocked()
    assert await primary.get_cloud_device_details() == ["primary"]
    assert member._client.get_gateway_devices.await_count == 1


async def test_pool_block_survives_restart(hass: HomeAssistant) -> None:
    """A saved block is read back when the API object is made again."""
    await async_load_blocks(hass)
    member = _api(hass, "member", 150)
    member.block(3600)
    assert "member" in hass.data[DOMAIN]["blocks"]
    again = _api(hass, "member", 150)
    assert again.is_blocked()
    again.clear_block()
    assert "member" not in hass.data[DOMAIN]["blocks"]


async def test_pool_member_devices_removed(hass: HomeAssistant) -> None:
    """Old devices of a pool member go, the primary devices stay."""
    primary_entry = MockConfigEntry(domain=DOMAIN, entry_id="primary")
    member_entry = MockConfigEntry(domain=DOMAIN, entry_id="member")
    primary_entry.add_to_hass(hass)
    member_entry.add_to_hass(hass)
    registry = dr.async_get(hass)
    registry.async_get_or_create(config_entry_id="primary", identifiers={(DOMAIN, "dev")})
    registry.async_get_or_create(config_entry_id="member", identifiers={(DOMAIN, "dev_2")})

    _remove_pool_member_devices(hass, member_entry)

    assert dr.async_entries_for_config_entry(registry, "member") == []
    assert len(dr.async_entries_for_config_entry(registry, "primary")) == 1
