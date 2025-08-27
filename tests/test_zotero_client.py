"""Tests for Zotero client module."""

import pytest
from unittest.mock import patch
from groundero.zotero_client import ZoteroClient


class TestZoteroClient:
    """Test cases for ZoteroClient."""

    def test_init_local(self):
        """Test initialization with local Zotero."""
        with patch.dict("os.environ", {"ZOTERO_LOCAL": "true"}):
            client = ZoteroClient()
            assert client.is_local is True
            assert client.zotero_client is None

    def test_init_remote_missing_credentials(self):
        """Test initialization with remote Zotero but missing credentials."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="library_id and api_key are required"):
                ZoteroClient()

    @patch("groundero.zotero_client.zotero.Zotero")
    def test_init_remote_with_credentials(self, mock_zotero):
        """Test initialization with remote Zotero and valid credentials."""
        with patch.dict(
            "os.environ", {"ZOTERO_LIBRARY_ID": "12345", "ZOTERO_API_KEY": "test_key"}
        ):
            client = ZoteroClient()
            assert client.is_local is False
            mock_zotero.assert_called_once_with("12345", "user", "test_key")

    def test_str_representation(self):
        """Test string representation of ZoteroClient."""
        with patch.dict("os.environ", {"ZOTERO_LOCAL": "true"}):
            client = ZoteroClient()
            result = str(client)
            assert "ZoteroClient" in result
            assert "local" in result.lower()
