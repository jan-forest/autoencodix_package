"""LLM client implementations."""

import os
import ollama
from dotenv import find_dotenv, load_dotenv
from mistralai import Mistral
from typing import List


PROMPT = """
You are a bioinformatics expert.

I performed differential gene expression analysis comparing healthy vs disease samples.

Below is the list of top altered genes. For each gene, interpret what the combined pattern suggests biologically.

Please provide:
1. A short explanation of the dominant biological themes.
2. 1–3 mechanistic hypotheses about what is happening in the diseased cells.
3. A summary of which pathways or processes may be affected.
4. If the gene list is ambiguous, state the reasonable alternatives.

Do NOT invent genes. Only use the information provided.

Here are the genes:
{gene_block}

Answer:
"""


class LLMExplainer:
    """LLM client with support for multiple providers."""

    def __init__(
        self,
        client_name: str,
        model_name: str,
        gene_list: List[str],
        prompt: str = PROMPT,
    ):
        """Initialize LLM client.

        Args:
            client_name: Name of the LLM client.
            model_name: Name of the model to use.
        """
        load_dotenv(find_dotenv())
        self._client_name = client_name
        self._model = model_name
        self.prompt = self._build_prompt(gene_list=gene_list, prompt=prompt)

        if self._client_name == "ollama":
            # Set Ollama host for Docker compatibility
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

            # Configure ollama client
            ollama_client = ollama.Client(host=ollama_host)
            self._ollama_client = ollama_client

            try:
                response = self._ollama_client.list()
                available_models = [m.model for m in response.models]
                if self._model not in available_models:
                    raise ValueError(
                        f"Model '{self._model}' not available. "
                        f"Available: {available_models}"
                    )
            except Exception as e:
                import warnings

                warnings.warn(f"Could not validate Ollama model '{self._model}': {e}")

        # Initialize client-specific objects
        elif self._client_name == "mistral":
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("Environment variable MISTRAL_API_KEY not set")
            self._mistral_client = Mistral(api_key=api_key)

    def _build_prompt(self, *, gene_list: List[str], prompt: str) -> str:
        gene_block = "\n".join(f"- {g}" for g in gene_list)
        return prompt.format(gene_block=gene_block)

    def explain(self) -> str:
        """Generate a response to a question.

        Args:
            question: The input question.

        Returns:
            Generated response text.

        Raises:
            ValueError: If temperature level is invalid.
        """
        print(self.prompt)
        if self._client_name == "mistral":
            return self._get_mistral_answer(question=self.prompt)
        elif self._client_name == "ollama":
            return self._get_ollama_answer(question=self.prompt)
        else:
            raise NotImplementedError(f"Client {self._client_name} not implemented")

    def _get_mistral_answer(self, *, question: str) -> str:
        """Get answer from Mistral API.

        Args:
            question: The input question.

        Returns:
            Generated response text.
        """
        chat_response = self._mistral_client.chat.complete(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": question,  # type: ignore
                },
            ],
        )
        return chat_response.choices[0].message.content  # type: ignore

    def _get_ollama_answer(self, *, question: str) -> str:
        """Get answer from Ollama.

        Args:
            question: The input question.

        Returns:
            Generated response text.
        """
        response = self._ollama_client.generate(
            model=self._model,
            prompt=question,
        )
        return response["response"]
