import os
import re
import json
import ollama
from dotenv import find_dotenv, load_dotenv
from mistralai.client import Mistral
from typing import List, Dict, Any

import requests
import warnings

warnings.simplefilter("always", UserWarning)


class LLMExplainer:
    """LLM client with support for multiple providers."""

    def __init__(
        self,
        client_name: str,
        model_name: str,
        genes_to_latent: Dict[str, List],
        prompt: str,
    ):
        """Initialize LLM client.

        Args:
            client_name: Name of the LLM client.
            model_name: Name of the model to use.
        """
        load_dotenv(find_dotenv())
        self._client_name = client_name
        self._model = model_name
        self.prompt = prompt
        self.genes_to_latent = genes_to_latent
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
        elif self._client_name == "openrouter":
            self._openrouter_api_key = os.environ.get("OPENROUTER_PREMIUM_API_KEY")
            if not self._openrouter_api_key:
                raise ValueError(
                    "Environment variable OPENROUTER_PREMIUM_API_KEY not set"
                )
            self._openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        elif self._client_name == "scads-llm":
            self._scads_llm_api_key = os.environ.get("SCADS_LLM_API_KEY")
            if not self._scads_llm_api_key:
                raise ValueError("Environment variable SCADS_LLM_API_KEY not set")
            self._scads_llm_url = "https://llm.scads.ai/v1"
        else:
            raise NotImplementedError(f"Client {self._client_name} not implemented")

    def _build_prompt(self, *, gene_list: List[str], prompt: str) -> str:
        """Builds the prompt for the LLM.

        Args:
            gene_list: List of genes.
            prompt: The prompt template.

        Returns:
            The formatted prompt.
        """
        gene_block = "\n".join(f"- {g}" for g in gene_list)
        return prompt.format(gene_block=gene_block)

    def extract_json_from_output(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM output robustly.

        Args:
            text: LLM output string.

        Returns:
            Parsed dict with keys "TLDR" and "DETAILS".
        """
        # Strip whitespace and common wrappers
        text = text.strip()
        # Remove code block markers if present
        text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.DOTALL)
        # Extract the JSON substring if embedded in text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        # Fix common issues: single quotes to double, trailing commas
        text = text.replace("'", '"')
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Remove unescaped newlines in strings (approximate fix)
        text = re.sub(r"(?<!\\)\n", " ", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            import warnings

            warnings.warn(f"Failed to parse JSON. Returning raw text. Error: {e}")
            return {
                "TLDR": text,
                "DETAILS": {
                    "dominant_themes": text,
                    "hypotheses": [],
                    "pathways_summary": text,
                },
            }

    def explain(self) -> Dict[str, Dict[str, Any]]:
        """Generate explanations for each latent dimension.

        Returns:
            A dict mapping latent dimension -> parsed JSON explanation (or raw output if parsing fails).
        """
        res: Dict[str, Dict[str, Any]] = {}
        markdown_sections = []
        for key, genes in self.genes_to_latent.items():
            print(f"Explaining latent dimension {key} with LLM...")
            prompt = self._build_prompt(gene_list=genes, prompt=self.prompt)
            raw_output = self._get_llm_answer(question=prompt)
            if len(raw_output) == 0:
                import warnings

                warnings.warn(
                    f"Received empty response from LLM for latent dimension {key}. Skipping."
                )
                continue
            # ---- TRY TO PARSE JSON ----
            try:
                parsed = self.extract_json_from_output(raw_output)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to parse JSON for latent dimension {key}. "
                    f"Using raw output instead. Error: {e}"
                )
                parsed = {
                    "TLDR": raw_output,
                    "DETAILS": {
                        "dominant_themes": raw_output,
                        "hypotheses": [],
                        "pathways_summary": raw_output,
                    },
                }
            res[key] = parsed
            # ---- BUILD READABLE MARKDOWN ----
            markdown_sections.append(f"# Latent Dimension {key}\n")
            markdown_sections.append("## Most influential genes")
            markdown_sections.append(", ".join(genes) + "\n")
            markdown_sections.append("## TLDR")
            markdown_sections.append(parsed.get("TLDR", "") + "\n")
            markdown_sections.append("## Details\n")

            details = parsed.get("DETAILS", {})
            if isinstance(details, dict):
                # Dominant themes
                markdown_sections.append("### Dominant Biological Themes\n")
                markdown_sections.append(details.get("dominant_themes", "") + "\n")
                # Hypotheses
                markdown_sections.append("### Mechanistic Hypotheses\n")
                hyps = details.get("hypotheses", [])
                if isinstance(hyps, list):
                    for hyp in hyps:
                        markdown_sections.append(f"- {hyp}\n")
                else:
                    markdown_sections.append(str(hyps) + "\n")
                # Pathways summary
                markdown_sections.append("### Summary of Pathways/Processes\n")
                markdown_sections.append(details.get("pathways_summary", "") + "\n")
            else:
                # Fallback if not dict
                markdown_sections.append(str(details) + "\n")

            markdown_sections.append("\n---\n")
        # ---- WRITE ONE SINGLE FILE ----
        output_path = os.path.join(os.getcwd(), "latent_explanations.md")
        with open(output_path, "w") as f:
            f.write("\n".join(markdown_sections))
        print(f"Saved explanations to: {output_path}")
        return res

    def _get_llm_answer(self, *, question: str) -> str:
        """Gets the LLM answer based on the client.

        Args:
            question: The input question.

        Returns:
            Generated response text.
        """
        if self._client_name == "mistral":
            return self._get_mistral_answer(question=question)
        elif self._client_name == "ollama":
            return self._get_ollama_answer(question=question)
        elif self._client_name == "openrouter":
            return self._get_openrouter_answer(
                question=question, model_name=self._model
            )
        elif self._client_name == "scads-llm":
            return self._get_scads_llm_answer(question=question, model_name=self._model)
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
            response_format={"type": "json_object"},  # Enforce JSON output
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

    def _get_openrouter_answer(self, *, question: str, model_name: str) -> str:
        """Get answer from OpenRouter.

        Args:
            question: The input question.
            model_name: The name of the model to use.

        Returns:
            Generated response text.
        """

        response = requests.post(
            url=self._openrouter_url,
            headers={
                "Authorization": f"Bearer {self._openrouter_api_key}",
            },
            data=json.dumps(
                {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                }
            ),
        )
        if response.status_code != 200:
            # Warn
            warnings.warn(
                f"OpenRouter API request failed with status code {response.status_code}: {response.text}"
            )
            # Return empty response
            return ""
        if "choices" not in response.json() or len(response.json()["choices"]) == 0:
            warnings.warn(
                f"OpenRouter API response missing 'choices': {response.json()}"
            )
            return ""
        return response.json()["choices"][0]["message"]["content"]

    def _get_scads_llm_answer(self, *, question: str, model_name: str) -> str:
        """Get answer from SCADS-LLM.

        Args:
            question: The input question.
            model_name: The name of the model to use.

        Returns:
            Generated response text.
        """

        client = OpenAI(
            base_url="https://llm.scads.ai/v1", api_key=self._scads_llm_api_key
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": question}], model=model_name
        )

        return response.choices[0].message.content
