import pydantic_ai.models
from pydantic_ai import Agent
import os
import csv
import json
import time
import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal
from pathlib import Path
import xml.etree.ElementTree as ET
import asyncio
import aiofiles

# Configuration
class Config:
    RESULTS_DIR = Path("llm_results")
    LOGS_DIR = Path("llm_logs")
    CSV_PATH = Path("models_hypotheses/combined_hypotheses.csv")
    MODELS = [
        "anthropic:claude-3-7-sonnet-latest",
        "openai:o3-mini",
        "openai:o4-mini"
    ]
    RETRY_ATTEMPTS = 5
    RETRY_DELAY = 5  # seconds
    MAX_CONCURRENT_EVALUATIONS = 10

# Initialize directories
Config.RESULTS_DIR.mkdir(exist_ok=True)
Config.LOGS_DIR.mkdir(exist_ok=True)


SYS_PROMPT = """
<SYSTEM>  
You are an expert biomedical reasoning engine, designed to assess scientific hypotheses with precision. Your task is to evaluate hypotheses about cancer causation based on novelty and plausibility, strictly adhering to structured output requirements.  
</SYSTEM>  
"""

PROMPT = """
<TASK>  
Assess the given hypothesis regarding factors contributing to a specific cancer.  
</TASK>  

<INPUT>  
<HYPOTHESIS> 
FACTORS: [HYPO] 
OUTCOME: predicted cancer type is [CANCER_TYPE]
</HYPOTHESIS>  
</INPUT>  

<OUTPUT_FORMAT>  
Strictly return the evaluation in the following XML format, without any additional text, commentary, or explanation:  

<EVALUATION>  
    <NOVELTY> X </NOVELTY>  
    <PLAUSIBILITY> Y </PLAUSIBILITY>  
</EVALUATION>  

Where:  
- **X (Novelty Grade)** is an integer from 1 to 10, reflecting how unique and groundbreaking the hypothesis is compared to known scientific literature.  
- **Y (Plausibility Grade)** is an integer from 1 to 10, indicating how scientifically credible the hypothesis is based on biomedical mechanisms and existing research.  

</OUTPUT_FORMAT>  

<RULES>  
1. Your response **must** contain **only** the XML block defined above. No explanations, reasoning, justifications, or extra text are allowed.  
2. Ensure scores align with rigorous scientific evaluation, leveraging both foundational biomedical knowledge and emerging research.  
3. Your assessment must be grounded in scientific plausibility, not speculation.  
4. If the hypothesis is fundamentally flawed but has been proposed in some form in the literature, reflect this in the scores rather than commenting on it.  
5. If the hypothesis is highly novel but lacks scientific support, score novelty high but plausibility low accordingly.  
</RULES>  
"""

HYPO_PLACEHOLDER = "[HYPO]"
CANCER_TYPE_PLACEHOLDER = "[CANCER_TYPE]"

class AsyncLogger:
    """
    Asynchronous Logger Class
    """

    def __init__(self):
        self.log_file = Config.LOGS_DIR / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.lock = asyncio.Lock()

    async def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp and level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"

        # Print to console
        print(log_entry)

        # Write to log file with lock to prevent race conditions
        async with self.lock:
            try:
                async with aiofiles.open(self.log_file, 'a') as f:
                    await f.write(log_entry + "\n")
            except Exception as e:
                print(f"[ERROR] Failed to write to log file: {str(e)}")


class HypothesisEvaluator:
    def __init__(self):
        self.logger = AsyncLogger()
        self.agents = self._initialize_agents()
        self.results = []
        self.results_lock = asyncio.Lock()  # Lock for thread-safe results access
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_EVALUATIONS)

    def _sys_prompt(self) -> str:
        return SYS_PROMPT

    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize agents for each model."""
        agents = {}
        for model_name in Config.MODELS:
            try:
                _model = pydantic_ai.models.infer_model(model_name)
                if model_name.startswith('anthropic'):
                    _model.api_key = os.getenv('ANTHROPIC_API_KEY')
                elif model_name.startswith('openai'):
                    _model.api_key = os.getenv('OPENAI_API_KEY')
                agents[model_name] = Agent(_model)
                agents[model_name].system_prompt(self._sys_prompt)
            except Exception as e:
                print(f"Failed to initialize agent for model {model_name}: {str(e)}")
        return agents

    async def load_hypotheses(self) -> List[Dict]:
        """Load hypotheses from CSV file."""
        try:
            df = pd.read_csv(Config.CSV_PATH)
            required_columns = ["hypo_id", "hypo_factors", "cancer_type"]

            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                await self.logger.log(f"CSV is missing required columns: {missing}", level="ERROR")
                return []

            hypotheses = df.to_dict('records')
            await self.logger.log(f"Loaded {len(hypotheses)} hypotheses from {Config.CSV_PATH}")
            return hypotheses
        except Exception as e:
            await self.logger.log(f"Error loading hypotheses from CSV: {str(e)}", level="ERROR")
            return []

    @staticmethod
    def prepare_prompt(hypothesis_text: str, cancer_type: str) -> str:
        """Replace [HYPO] and [CANCER_TYPE] placeholder with actual hypothesis."""
        curr_prompt = PROMPT.replace(HYPO_PLACEHOLDER, hypothesis_text)
        curr_prompt = curr_prompt.replace(CANCER_TYPE_PLACEHOLDER, cancer_type)
        return curr_prompt

    async def evaluate_single_model(self, hypothesis_id: str, hypothesis_text: str,
                                    cancer_type: str, model_name: str, agent: Agent) -> None:
        """Evaluate a hypothesis with a single model."""
        prepared_prompt = self.prepare_prompt(hypothesis_text, cancer_type)

        await self.logger.log(f"Evaluating hypothesis {hypothesis_id} with model {model_name}")

        for attempt in range(Config.RETRY_ATTEMPTS):
            try:
                # Use run instead of run_sync to get an awaitable
                response = await agent.run(prepared_prompt)
                result = self._parse_evaluation(response.data)

                if result:
                    # Store the result with thread safety
                    async with self.results_lock:
                        evaluation_result = {
                            "hypothesis_id": hypothesis_id,
                            "model": model_name,
                            "novelty": result["novelty"],
                            "plausibility": result["plausibility"],
                            "raw_response": response.data,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.results.append(evaluation_result)

                    await self.logger.log(f"Successfully evaluated hypothesis {hypothesis_id} with {model_name}: "
                                          f"Novelty={result['novelty']}, Plausibility={result['plausibility']}")
                    break
                else:
                    await self.logger.log(
                        f"Failed to parse evaluation from {model_name} response for hypothesis {hypothesis_id} "
                        f"(Attempt {attempt + 1}/{Config.RETRY_ATTEMPTS})", level="WARNING")

            except Exception as e:
                await self.logger.log(f"Error evaluating hypothesis {hypothesis_id} with {model_name}: {str(e)} "
                                      f"(Attempt {attempt + 1}/{Config.RETRY_ATTEMPTS})", level="ERROR")

            if attempt < Config.RETRY_ATTEMPTS - 1:
                await asyncio.sleep(Config.RETRY_DELAY)

    async def evaluate_hypothesis(self, hypothesis_id: str, hypothesis_text: str, cancer_type: str) -> None:
        """Evaluate a hypothesis with all configured models concurrently."""
        # Use semaphore to limit concurrency
        async with self.semaphore:
            # Create tasks for each model evaluation
            tasks = []
            for model_name, agent in self.agents.items():
                task = asyncio.create_task(
                    self.evaluate_single_model(hypothesis_id, hypothesis_text, cancer_type, model_name, agent)
                )
                tasks.append(task)

            # Wait for all model evaluations to complete
            await asyncio.gather(*tasks)

    def _parse_evaluation(self, response_text: str) -> Optional[Dict[str, int]]:
        """Parse novelty and plausibility scores from XML response."""
        try:
            # Remove any non-XML content that might be present
            xml_match = re.search(r'<EVALUATION>.*?</EVALUATION>', response_text, re.DOTALL)
            if not xml_match:
                return None

            xml_content = xml_match.group(0)
            root = ET.fromstring(xml_content)

            novelty = int(root.find('NOVELTY').text.strip())
            plausibility = int(root.find('PLAUSIBILITY').text.strip())

            return {
                "novelty": novelty,
                "plausibility": plausibility
            }
        except Exception as e:
            print(f"Error parsing evaluation: {str(e)}")
            return None

    async def save_results(self) -> None:
        """Save all evaluation results to CSV and JSON."""
        if not self.results:
            await self.logger.log("No results to save", level="WARNING")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as CSV
        csv_path = Config.RESULTS_DIR / f"evaluations_{timestamp}.csv"
        try:
            df = pd.DataFrame(self.results)
            df_summary = df[["hypothesis_id", "model", "novelty", "plausibility", "timestamp"]]
            df_summary.to_csv(csv_path, index=False)
            await self.logger.log(f"Saved results to CSV: {csv_path}")
        except Exception as e:
            await self.logger.log(f"Error saving CSV results: {str(e)}", level="ERROR")

        # Save as JSON (with full raw responses)
        json_path = Config.RESULTS_DIR / f"evaluations_{timestamp}.json"
        try:
            async with aiofiles.open(json_path, 'w') as f:
                await f.write(json.dumps(self.results, indent=2))
            await self.logger.log(f"Saved results to JSON: {json_path}")
        except Exception as e:
            await self.logger.log(f"Error saving JSON results: {str(e)}", level="ERROR")

        # Generate summary statistics
        await self._generate_summary(timestamp)

    async def _generate_summary(self, timestamp: str) -> None:
        """Generate and save summary statistics of evaluations."""
        try:
            df = pd.DataFrame(self.results)

            # Create summary by model
            model_summary = df.groupby('model').agg({
                'novelty': ['mean', 'min', 'max', 'std'],
                'plausibility': ['mean', 'min', 'max', 'std']
            }).round(2)

            # Create summary by hypothesis
            hypothesis_summary = df.groupby('hypothesis_id').agg({
                'novelty': ['mean', 'min', 'max', 'std'],
                'plausibility': ['mean', 'min', 'max', 'std']
            }).round(2)

            # Save summaries
            summary_path = Config.RESULTS_DIR / f"summary_{timestamp}.txt"
            summary_content = "=== EVALUATION SUMMARY ===\n\n"
            summary_content += "--- BY MODEL ---\n"
            summary_content += model_summary.to_string()
            summary_content += "\n\n--- BY HYPOTHESIS ---\n"
            summary_content += hypothesis_summary.to_string()

            async with aiofiles.open(summary_path, 'w') as f:
                await f.write(summary_content)

            await self.logger.log(f"Generated summary statistics: {summary_path}")
        except Exception as e:
            await self.logger.log(f"Error generating summary: {str(e)}", level="ERROR")


async def process_batch(evaluator, batch):
    """Process a batch of hypotheses concurrently."""
    tasks = []
    for hypothesis in batch:
        hypothesis_id = hypothesis["hypo_id"]
        hypothesis_text = hypothesis["hypo_factors"]
        cancer_type = hypothesis["cancer_type"]
        task = asyncio.create_task(
            evaluator.evaluate_hypothesis(hypothesis_id, hypothesis_text, cancer_type)
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


async def main_async():
    """Asynchronous main execution function."""
    logger = AsyncLogger()
    await logger.log("Starting hypothesis evaluation process")

    # Initialize evaluator
    evaluator = HypothesisEvaluator()

    # Load hypotheses
    hypotheses = await evaluator.load_hypotheses()
    if not hypotheses:
        await logger.log("No hypotheses to evaluate. Exiting.", level="ERROR")
        return

    # Process hypotheses in batches for better control
    batch_size = Config.MAX_CONCURRENT_EVALUATIONS
    for i in range(0, len(hypotheses), batch_size):
        batch = hypotheses[i:i + batch_size]
        await logger.log(f"Processing batch {i // batch_size + 1} of {(len(hypotheses) - 1) // batch_size + 1}")
        await process_batch(evaluator, batch)

    # Save all results
    await evaluator.save_results()
    await logger.log("Hypothesis evaluation process completed")


def main():
    """Wrapper for the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()