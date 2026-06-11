"""Integrations: connect the assistant to outside services.

Each integration declares the credentials it needs and reports its connection
status. Token-based services (GitHub, Vercel, Discord, local Notes) work as
soon as you provide a token. OAuth services (Google, Microsoft) and the social
APIs work once you supply an access token from their developer console.

Nothing here transmits data unless an integration is configured and used.
"""

from ..config import Config
from .base import Integration, IntegrationRegistry
from .devtools import GitHubIntegration, VSCodeIntegration, VercelIntegration
from .google import (
    GoogleDocsIntegration,
    GoogleDriveIntegration,
    GmailIntegration,
    GoogleMeetIntegration,
)
from .microsoft import (
    OneDriveIntegration,
    OutlookIntegration,
    TeamsIntegration,
    WordIntegration,
)
from .notes import LocalNotesIntegration
from .social import DiscordIntegration, InstagramIntegration, TwitterIntegration


def build_integrations(config: Config) -> IntegrationRegistry:
    """Assemble the integration panel."""
    reg = IntegrationRegistry()
    # Google
    reg.register(GmailIntegration(config))
    reg.register(GoogleDriveIntegration(config))
    reg.register(GoogleDocsIntegration(config))
    reg.register(GoogleMeetIntegration(config))
    # Microsoft
    reg.register(OutlookIntegration(config))
    reg.register(OneDriveIntegration(config))
    reg.register(WordIntegration(config))
    reg.register(TeamsIntegration(config))
    # Social
    reg.register(TwitterIntegration(config))
    reg.register(InstagramIntegration(config))
    reg.register(DiscordIntegration(config))
    # Dev tools
    reg.register(GitHubIntegration(config))
    reg.register(VSCodeIntegration(config))
    reg.register(VercelIntegration(config))
    # Notes
    reg.register(LocalNotesIntegration(config))
    return reg


__all__ = ["Integration", "IntegrationRegistry", "build_integrations"]
