"""Platform for the Daikin AC."""

import logging

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady, OAuth2TokenRequestError, OAuth2TokenRequestReauthError
from homeassistant.helpers import config_entry_oauth2_flow, device_registry as dr
from homeassistant.helpers.config_entry_oauth2_flow import ImplementationUnavailableError
from homeassistant.helpers.typing import ConfigType

from daikin_onecta.auth import get_account_id
from daikin_onecta.exceptions import OnectaAccessTokenError

from .binary_sensor import migrate_legacy_binary_sensor_unique_ids
from .const import CONF_POOL_MEMBER, DOMAIN
from .coordinator import OnectaDataUpdateCoordinator
from .daikin_api import DaikinApi, async_load_blocks
from .device import migrate_legacy_entity_unique_ids, migrate_legacy_subdevice_identifiers
from .sensor import migrate_legacy_sensor_unique_ids
from .update import migrate_legacy_update_unique_ids

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [
    Platform.CLIMATE,
    Platform.FAN,
    Platform.SENSOR,
    Platform.WATER_HEATER,
    Platform.SWITCH,
    Platform.SELECT,
    Platform.BINARY_SENSOR,
    Platform.BUTTON,
    Platform.UPDATE,
]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Daikin Onecta component."""
    return True


def _pool_registry(hass: HomeAssistant) -> dict[str, DaikinApi]:
    """Return the shared registry of pool member accounts."""
    return hass.data.setdefault(DOMAIN, {}).setdefault("pool", {})


def _remove_pool_member_devices(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    """Remove devices that a pool member made before it joined the pool.

    POOL: a pool member makes no entities, so its old devices stay empty.
    A device that belongs only to this entry is deleted. The devices of the
    primary entry are separate registry entries and stay.
    """
    registry = dr.async_get(hass)
    for device in dr.async_entries_for_config_entry(registry, config_entry.entry_id):
        _LOGGER.info("Removing device '%s' from Daikin pool member '%s'", device.name, config_entry.title)
        registry.async_update_device(device.id, remove_config_entry_id=config_entry.entry_id)


def is_pool_member(config_entry: ConfigEntry) -> bool:
    """Return whether this entry only lends its rate limit to the pool."""
    return config_entry.options.get(CONF_POOL_MEMBER, False) is True


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Establish connection with Daikin."""
    try:
        implementation = await config_entry_oauth2_flow.async_get_config_entry_implementation(hass, config_entry)
    except ImplementationUnavailableError as err:
        raise ConfigEntryNotReady(
            translation_domain=DOMAIN,
            translation_key="oauth2_implementation_unavailable",
        ) from err

    await async_load_blocks(hass)
    pool = _pool_registry(hass)
    daikin_api = DaikinApi(hass, config_entry, implementation, pool)

    try:
        await daikin_api.async_get_access_token()
    except OAuth2TokenRequestReauthError as err:
        raise ConfigEntryAuthFailed from err
    except (OAuth2TokenRequestError, aiohttp.ClientError) as err:
        raise ConfigEntryNotReady from err

    if is_pool_member(config_entry):
        # POOL: register the account and stop. No coordinator, no platforms.
        _LOGGER.info("Daikin entry '%s' set up as API pool member", config_entry.title)
        pool[config_entry.entry_id] = daikin_api
        _remove_pool_member_devices(hass, config_entry)
        config_entry.runtime_data = None
        config_entry.async_on_unload(config_entry.add_update_listener(update_listener))
        return True

    config_entry.runtime_data = OnectaDataUpdateCoordinator(hass, config_entry, daikin_api)

    await config_entry.runtime_data.async_config_entry_first_refresh()
    migrate_legacy_subdevice_identifiers(hass, config_entry, config_entry.runtime_data.data or {})
    migrate_legacy_entity_unique_ids(hass, config_entry, config_entry.runtime_data.data or {})
    migrate_legacy_binary_sensor_unique_ids(hass, config_entry, config_entry.runtime_data.data or {})
    migrate_legacy_sensor_unique_ids(hass, config_entry, config_entry.runtime_data.data or {})
    migrate_legacy_update_unique_ids(hass, config_entry)

    config_entry.async_on_unload(config_entry.add_update_listener(update_listener))

    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.debug("Unloading integration")
    if config_entry.runtime_data is None:
        _pool_registry(hass).pop(config_entry.entry_id, None)
        return True
    return await hass.config_entries.async_unload_platforms(config_entry, PLATFORMS)


async def update_listener(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    """Handle options update."""
    # A pool member has no coordinator. Reload when the role changes.
    if (config_entry.runtime_data is None) != is_pool_member(config_entry):
        hass.config_entries.async_schedule_reload(config_entry.entry_id)
        return
    if config_entry.runtime_data is None:
        return
    coordinator: OnectaDataUpdateCoordinator = config_entry.runtime_data
    coordinator.update_settings(config_entry)
    coordinator.async_update_listeners()


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.info("Migration from version %s.%s", config_entry.version, config_entry.minor_version)

    if config_entry.version == 1:
        match config_entry.minor_version:
            case 1:
                try:
                    unique_id = get_account_id(config_entry.data["token"]["access_token"])
                except (KeyError, OnectaAccessTokenError):
                    _LOGGER.exception("Failed to read the account ID from the OAuth token during migration")
                    return False
                hass.config_entries.async_update_entry(
                    config_entry,
                    minor_version=2,
                    unique_id=unique_id,
                )

    _LOGGER.info("Migration to version %s.%s successful", config_entry.version, config_entry.minor_version)
    return True
