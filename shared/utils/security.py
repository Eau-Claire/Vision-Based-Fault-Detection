import socket
import ipaddress
from urllib.parse import urlparse
from typing import List, Optional

def is_safe_url(
    url: str, 
    allowed_hosts: Optional[List[str]] = None, 
    allow_private_ips: bool = False
) -> bool:
    """Validate URL for safety, preventing SSRF and DNS rebinding attacks.
    
    Args:
        url: The URL to validate.
        allowed_hosts: Optional list of hostnames/domains that are allowed.
        allow_private_ips: If True, private/loopback/link-local IPs are allowed (for local dev).
    
    Returns:
        True if the URL is safe, False otherwise.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
            
        hostname = parsed.hostname
        if not hostname:
            return False
            
        # Check allowed hosts if provided
        if allowed_hosts:
            hostname_lower = hostname.lower()
            allowed_hosts_lower = [h.lower() for h in allowed_hosts]
            if hostname_lower not in allowed_hosts_lower:
                return False
                
        # If we allow private IPs (local development), we skip IP-level resolution check
        if allow_private_ips:
            return True
            
        # Resolve hostname to IP addresses and check each to prevent SSRF
        ips = socket.getaddrinfo(hostname, None)
        for ip_info in ips:
            ip_str = ip_info[4][0]
            # Handle scoped IPv6 addresses (e.g. fe80::1%lo0)
            if "%" in ip_str:
                ip_str = ip_str.split("%")[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
                
        return True
    except Exception:
        return False
