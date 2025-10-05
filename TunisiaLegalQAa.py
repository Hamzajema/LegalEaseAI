"""
Tunisia Legal Q&A System - Fully Local Version
A module to process legal PDFs and answer questions using local LLMs.
No API keys required - everything runs on your machine.
"""

import os
import pickle
import json
from typing import List, Dict, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PDF processing
import PyPDF2
import pdfplumber

# Text processing and embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Local LLM
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
import torch


class TunisiaLegalQA:
    """
    A fully local question-answering system for Tunisian legal documents.
    Uses open-source models that run on your machine.
    """
    
    def __init__(
        self, 
        pdf_directory: str, 
        cache_dir: str = "./cache",
        model_name: str = "mistral-7b-instruct",  # or "llama2", "phi"
        use_gpu: bool = True
    ):
        """
        Initialize the Q&A system.
        
        Args:
            pdf_directory: Path to directory containing PDF files
            cache_dir: Directory to store processed data and models
            model_name: Local LLM to use ('mistral-7b-instruct', 'llama2-7b', 'phi-2')
            use_gpu: Whether to use GPU if available
        """
        self.pdf_directory = Path(pdf_directory)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize embedding model (lightweight, runs on CPU)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Storage for documents
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # LLM will be loaded on demand
        self.llm_pipeline = None
        self.model_name = model_name
        
    def load_llm(self):
        """
        Load the local LLM model.
        This is done separately so embeddings can work without loading heavy model.
        """
        if self.llm_pipeline is not None:
            return
        
        print(f"Loading local LLM model: {self.model_name}...")
        
        # Model mapping
        model_map = {
            "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
            "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
            "phi-2": "microsoft/phi-2",
            "tiny-llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta"
        }
        
        model_id = model_map.get(self.model_name, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        try:
            # Use 4-bit quantization to reduce memory usage
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=model_id,
                    tokenizer=model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    model_kwargs={"quantization_config": quantization_config},
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                )
            else:
                # CPU version
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=model_id,
                    tokenizer=model_id,
                    device=self.device,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                )
            
            print("LLM loaded successfully!")
            
        except Exception as e:
            print(f"Error loading {model_id}: {e}")
            print("Falling back to TinyLlama...")
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device=self.device,
                max_new_tokens=512,
            )
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, str]]:
        """
        Extract text from a PDF file, page by page.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page text and metadata
        """
        chunks = []
        
        try:
            # Try pdfplumber first (better for complex PDFs)
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            chunks.append({
                                'text': text.strip(),
                                'source': pdf_path.name,
                                'page': page_num
                            })
                        
                        # Progress indicator
                        if page_num % 10 == 0:
                            print(f"  Processed {page_num}/{total_pages} pages...")
                    except Exception as e:
                        print(f"  Error on page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path.name}, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            text = page.extract_text()
                            if text and text.strip():
                                chunks.append({
                                    'text': text.strip(),
                                    'source': pdf_path.name,
                                    'page': page_num
                                })
                            
                            # Progress indicator
                            if page_num % 10 == 0:
                                print(f"  Processed {page_num}/{total_pages} pages...")
                        except Exception as e:
                            print(f"  Error on page {page_num}: {e}")
                            continue
                            
            except Exception as e2:
                print(f"Error extracting text from {pdf_path.name}: {e2}")
        
        print(f"  Extracted {len(chunks)} pages from {pdf_path.name}")
        return chunks
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        # Safety check
        if overlap >= chunk_size:
            overlap = chunk_size // 2
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            
            # Try to break at sentence boundaries (only if not at the end)
            if end < text_length and len(chunk) > chunk_size * 0.5:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.3:  # At least 30% of chunk size
                    end = start + break_point + 1
                    chunk = text[start:end]
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
            
            # Move forward, ensuring we make progress
            if end - start <= overlap:
                start = end  # Prevent infinite loop
            else:
                start = end - overlap
            
            # Safety break
            if len(chunks) > 10000:  # Reasonable limit
                print(f"Warning: Reached chunk limit for safety. Text may be truncated.")
                break
            
        return chunks
    
    def process_pdfs(self, force_reprocess: bool = False):
        """
        Process all PDFs in the directory and create embeddings.
        
        Args:
            force_reprocess: If True, reprocess even if cache exists
        """
        cache_file = self.cache_dir / "processed_data.pkl"
        
        # Check if cache exists
        if cache_file.exists() and not force_reprocess:
            print("Loading from cache...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
                self.metadata = data['metadata']
            print(f"Loaded {len(self.documents)} document chunks from cache.")
            
            # If cache is empty, force reprocess
            if len(self.documents) == 0:
                print("Cache is empty, forcing reprocess...")
                force_reprocess = True
            else:
                return
        
        if not force_reprocess:
            force_reprocess = True
        
        print("Processing PDFs...")
        all_chunks = []
        
        # Check if directory exists
        if not self.pdf_directory.exists():
            raise FileNotFoundError(
                f"PDF directory not found: {self.pdf_directory}\n"
                f"Please create the directory and add your PDF files."
            )
        
        # Process each PDF
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        if len(pdf_files) == 0:
            raise FileNotFoundError(
                f"No PDF files found in: {self.pdf_directory}\n"
                f"Please add PDF files to this directory."
            )
        
        print(f"Found {len(pdf_files)} PDF files.")
        
        for pdf_path in pdf_files:
            print(f"\nProcessing {pdf_path.name}...")
            try:
                pages = self.extract_text_from_pdf(pdf_path)
                
                # Further chunk each page if needed
                page_count = 0
                for page_data in pages:
                    try:
                        text_chunks = self.chunk_text(page_data['text'])
                        for i, chunk in enumerate(text_chunks):
                            if len(chunk) > 50:  # Only keep meaningful chunks
                                all_chunks.append({
                                    'text': chunk,
                                    'source': page_data['source'],
                                    'page': page_data['page'],
                                    'chunk_id': i
                                })
                        page_count += 1
                    except Exception as e:
                        print(f"  Error chunking page {page_data['page']}: {e}")
                        continue
                
                print(f"  Successfully processed {page_count} pages")
                
            except Exception as e:
                print(f"  Failed to process {pdf_path.name}: {e}")
                continue
        
        print(f"Created {len(all_chunks)} text chunks.")
        
        # Extract texts and metadata
        self.documents = [chunk['text'] for chunk in all_chunks]
        self.metadata = [{k: v for k, v in chunk.items() if k != 'text'} 
                        for chunk in all_chunks]
        
        # Create embeddings
        print("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(
            self.documents, 
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Save to cache
        print("Saving to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }, f)
        
        print("Processing complete!")
        print(f"Total chunks: {len(self.documents)}")
        print(f"Cache saved to: {cache_file}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents given a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        if self.embeddings is None or len(self.documents) == 0:
            raise ValueError("No documents processed. Call process_pdfs() first.")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Ensure embeddings is 2D array
        if len(self.embeddings.shape) == 1:
            self.embeddings = self.embeddings.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'similarity': float(similarities[idx]),
                'source': self.metadata[idx]['source'],
                'page': self.metadata[idx]['page'],
                'chunk_id': self.metadata[idx]['chunk_id']
            })
        
        return results
    
    def answer_question(
        self, 
        question: str, 
        top_k: int = 3,
        use_llm: bool = True,
        language: str = "en"
    ) -> Dict:
        """
        Answer a question using the processed documents and local LLM.
        
        Args:
            question: Question to answer
            top_k: Number of relevant chunks to consider
            use_llm: Whether to use LLM for answer generation
            language: Response language ('en', 'fr', 'ar')
            
        Returns:
            Dictionary containing answer and sources
        """
        # Search for relevant documents
        results = self.search(question, top_k=top_k)
        
        if not results:
            return {
                'question': question,
                'answer': "I couldn't find relevant information to answer this question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare sources
        sources = [{
            'source': r['source'],
            'page': r['page'],
            'similarity': r['similarity']
        } for r in results]
        
        # Generate answer
        if use_llm:
            # Load LLM if not already loaded
            if self.llm_pipeline is None:
                self.load_llm()
            
            answer = self._generate_llm_answer(question, results, language)
        else:
            answer = self._generate_simple_answer(question, results)
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'confidence': float(results[0]['similarity'])
        }
    
    def _generate_simple_answer(self, question: str, results: List[Dict]) -> str:
        """
        Generate a simple answer without LLM (just return relevant text).
        """
        best_match = results[0]
        
        answer = (
            f"Based on {best_match['source']} (Page {best_match['page']}):\n\n"
            f"{best_match['text']}"
        )
        
        return answer
    
    def _generate_llm_answer(
        self, 
        question: str, 
        results: List[Dict],
        language: str = "en"
    ) -> str:
        """
        Generate an answer using the local LLM.
        """
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {i} - {result['source']}, Page {result['page']}]\n"
                f"{result['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Language-specific prompts
        if language == "fr":
            system_prompt = "Vous êtes un expert juridique sur le droit des affaires tunisien."
            instruction = "Répondez à la question suivante en vous basant sur le contexte fourni. Soyez clair et concis."
        elif language == "ar":
            system_prompt = "أنت خبير قانوني في قانون الأعمال التونسي."
            instruction = "أجب على السؤال التالي بناءً على السياق المقدم. كن واضحاً وموجزاً."
        else:
            system_prompt = "You are a legal expert on Tunisian business law and startup regulations."
            instruction = "Answer the following question based on the provided context. Be clear and concise."
        
        # Create prompt optimized for Phi-2
        prompt = f"""Instruct: {system_prompt}

{instruction}

Context:
{context}

Question: {question}

Output:"""
        
        # Generate answer
        try:
            output = self.llm_pipeline(
                prompt,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=50256
            )
            
            # Extract the generated text
            generated_text = output[0]['generated_text']
            
            # Remove the prompt from the output
            if "Output:" in generated_text:
                answer = generated_text.split("Output:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # Clean up common artifacts
            answer = answer.replace("[/INST]", "")
            answer = answer.replace("</s>", "")
            answer = answer.replace("<s>", "")
            
            # Remove "Solution 0:" prefix if present
            if answer.startswith("Solution 0:"):
                answer = answer.replace("Solution 0:", "").strip()
            
            # If answer is empty or very short, use simple answer
            if len(answer.strip()) < 20:
                return self._generate_simple_answer(question, results)
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer with LLM: {e}")
            return self._generate_simple_answer(question, results)
    
    def save_knowledge_base(self, output_file: str):
        """
        Save the processed knowledge base to a file for easy sharing.
        """
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'metadata': self.metadata
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Knowledge base saved to {output_file}")
    
    def interactive_mode(self):
        """
        Start an interactive Q&A session.
        """
        print("\n" + "="*80)
        print("Tunisia Legal Q&A System - Interactive Mode")
        print("="*80)
        print("Ask questions about Tunisian law, startups, and business.")
        print("Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n🔍 Searching...")
            result = self.answer_question(question, use_llm=True)
            
            print(f"\n💡 Answer:\n{result['answer']}\n")
            
            if result['sources']:
                print("📚 Sources:")
                for source in result['sources']:
                    print(f"  • {source['source']}, Page {source['page']} "
                          f"(Relevance: {source['similarity']:.2%})")
            
            print(f"\n✓ Confidence: {result['confidence']:.2%}")


# Example usage and CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tunisia Legal Q&A System - Local Version"
    )
    parser.add_argument(
        "--pdf-dir", 
        default="./tunisia_legal_pdfs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--process", 
        action="store_true",
        help="Process PDFs and create embeddings"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Start interactive Q&A mode"
    )
    parser.add_argument(
        "--question", 
        type=str,
        help="Ask a single question"
    )
    parser.add_argument(
        "--model",
        default="tiny-llama",
        choices=["mistral-7b-instruct", "llama2-7b", "phi-2", "tiny-llama", "zephyr-7b"],
        help="Local LLM model to use"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing Tunisia Legal Q&A System...")
    qa_system = TunisiaLegalQA(
        pdf_directory=args.pdf_dir,
        model_name=args.model,
        use_gpu=not args.no_gpu
    )
    
    # Process PDFs if requested
    if args.process:
        qa_system.process_pdfs(force_reprocess=True)
    else:
        qa_system.process_pdfs()
    
    # Interactive mode
    if args.interactive:
        qa_system.interactive_mode()
    
    # Single question
    elif args.question:
        result = qa_system.answer_question(args.question, use_llm=True)
        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources:")
        for source in result['sources']:
            print(f"  - {source['source']}, Page {source['page']}")
    
    # Default: show examples
    else:
        print("\nExample usage:")
        print("  python tunisia_legal_qa.py --process  # Process PDFs first")
        print("  python tunisia_legal_qa.py --interactive  # Start Q&A session")
        print("  python tunisia_legal_qa.py --question 'What is the Startup Act?'")