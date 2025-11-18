from smolagents import tool
import trafilatura
import requests

@tool
def read_links(links: list[str]) -> str:
    """
    Reads the main text content from a list of up to 3 URLs.
    Choose the three most important links to read to avoid overwhelming the context.

    Args:
        links: A list of URLs to read (maximum of 3).

    Returns:
        A string containing the combined text content of the websites.
    """
    if len(links) > 3:
        links = links[:3]
        
    all_text = ""
    for link in links:
        try:
            response = requests.get(link, timeout=10)
            if response.status_code == 200:
                text = trafilatura.extract(response.text)
                if text:
                    all_text += f"--- Content from {link} ---\n{text}\n\n"
            else:
                all_text += f"--- Content from {link} ---\nError: Received status code {response.status_code}\n\n"
        except Exception as e:
            all_text += f"--- Content from {link} ---\nError: {e}\n\n"
    return all_text

