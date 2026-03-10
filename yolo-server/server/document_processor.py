import re
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

class WebScraper:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_url(self, url: str) -> str:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return response.text
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            raise

    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, 'lxml')
        
        title = soup.title.string.strip() if soup.title else url
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        paragraphs = soup.find_all('p')
        content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        if not content:
            content = soup.get_text(separator='\n', strip=True)
        
        return {
            'title': title,
            'content': content,
            'url': url
        }

    def scrape(self, url: str) -> Dict[str, Any]:
        html = self.fetch_url(url)
        return self.extract_content(html, url)

class TextChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        overlap_percent: float = 0.1
    ):
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_percent)

    def _split_by_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[。！？.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> List[str]:
        sentences = self._split_by_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_length = len(sentence)

            if current_length + sentence_length <= self.chunk_size or not current_chunk:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
                i += 1
            else:
                chunks.append(' '.join(current_chunk))
                
                if self.overlap_size > 0:
                    overlap_text = ' '.join(current_chunk)
                    while len(overlap_text) > self.overlap_size and len(current_chunk) > 1:
                        current_chunk.pop(0)
                        overlap_text = ' '.join(current_chunk)
                    current_length = len(overlap_text) + 1
                else:
                    current_chunk = []
                    current_length = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_document(
        self,
        document: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        text = document.get('content', '')
        chunks = self.chunk_text(text)
        
        chunked_docs = []
        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                'content': chunk,
                'title': document.get('title', ''),
                'source': document.get('url', ''),
                'chunk_index': idx,
                'total_chunks': len(chunks)
            })
        
        return chunked_docs

class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 512,
        overlap_percent: float = 0.1
    ):
        self.scraper = WebScraper()
        self.chunker = TextChunker(chunk_size, overlap_percent)

    def process_url(self, url: str) -> List[Dict[str, Any]]:
        document = self.scraper.scrape(url)
        return self.chunker.chunk_document(document)

    def process_text(
        self,
        text: str,
        title: str = '',
        source: str = ''
    ) -> List[Dict[str, Any]]:
        document = {
            'content': text,
            'title': title,
            'url': source
        }
        return self.chunker.chunk_document(document)
