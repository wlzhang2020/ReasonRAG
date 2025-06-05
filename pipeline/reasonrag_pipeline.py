from __future__ import annotations
import asyncio
import time
import os
import pickle
import psutil
from copy import deepcopy
import os
from multiprocessing import Pool
import time
from tqdm import tqdm
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import re
import math
from tqdm import tqdm
import numpy as np
import logging
import copy
import queue
from typing import List
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from flashrag.utils import get_retriever, get_generator, selfask_pred_parse, ircot_pred_parse
from flashrag.pipeline import BasicPipeline
from flashrag.dataset import get_batch_dataset, merge_batch_dataset, Dataset
from flashrag.evaluator.metrics import F1_Score
from flashrag.prompt import PromptTemplate
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar, Union


def save_data(save_dir, data, file_name="intermediate_data.json"):
    """Save the evaluated data, including the raw data and the score of each data
    sample on each metric."""
    save_path = os.path.join(save_dir, file_name)
    data.save(save_path)

class MCTSRAGNode:
    def __init__(self, index, S: Any, parent: MCTSRAGNode | None = None, parent_id: int = -1, step: int = 0,
                 next_state: str | None = None, question="RAG question", thoughts=None, golden_answers=None,
                 node_dict: dict[str, Any] | None = None):
        self.id = index
        self.S = S
        self.parent = parent
        self.parent_id = parent_id
        self.children = []
        self.children_ids = []
        self.N = 0
        self.step = step
        self.Q = 0.0
        self.Q_list = []
        self.reward_list = []
        self.next_state = next_state
        self.question = question
        self.thoughts = thoughts
        self.golden_answers = golden_answers
        self.node_dict = node_dict if node_dict is not None else {}
        self.max_tokens_reached = False

    def add_child(self, child_node: MCTSRAGNode):
        self.children.append(child_node)
        self.children_ids.append(child_node.id)

    def update_Q(self, reward: float):
        self.N += 1
        self.Q_list.append(reward)
        self.Q = np.mean(self.Q_list)
        self.node_dict['Q'] = self.Q
        self.node_dict['step'] = self.step
        self.node_dict['N'] = self.N

    def update_reward(self, reward: float):
        self.reward_list.append(reward)
        self.node_dict["reward"] = np.mean(self.reward_list)

    def is_fully_expanded(self, max_children) -> bool:
        return self.next_state is None or len(self.children) == max_children


def extract_answer(pred, prefix="So the answer is"):
    if prefix in pred:
        pred = pred.split(prefix)[1].strip()
    answer_matches = re.findall(r'<answer>(.*?)</answer>', pred)
    pred = answer_matches[-1] if answer_matches else pred
    pred = re.sub(r'<answer.*?>.*?</answer>|<query.*?>.*?</query>|answer>|<answer', '', pred, flags=re.DOTALL)
    if '.' in pred:
        pred = pred.split('.')[0].strip()
    else:
        pred = pred.strip()

    return pred

def reason_rag_pred_parse(dataset):
    FINAL_ANSWER_PREFIX = "So the answer is"
    for item in dataset:
        pred = item.pred
        if FINAL_ANSWER_PREFIX in pred:
            answer = pred.split(FINAL_ANSWER_PREFIX)[1].strip()
        else:
            answer = pred
        answer_matches = re.findall(r'<answer.*?>([^<]*)</answer>', answer, re.DOTALL)
        answer = answer_matches[-1] if answer_matches else pred
        answer = re.sub(r'<answer.*?>.*?</answer>|<query.*?>.*?</query>|answer>|<answer', '', answer, flags=re.DOTALL)
        answer = re.sub(r'  +', ' ', answer)
        if '.' in answer:
            answer = answer.split('.')[0].strip()
        else:
            answer = answer.strip()

        item.update_output('raw_pred', pred)
        item.update_output('pred', answer)
    return dataset

def extract_last_number(s):
    matches = re.findall(r'[+-]?\d+', s)
    if matches:
        last_number = float(matches[-1])
        return last_number

    return None


def process_text(text):
    next_match = re.search(r'So the next (?:query )?is (.*?)(?= So the answer is|$)', text, re.DOTALL)
    if next_match:
        next_content = next_match.group(1).strip()
        next_content = re.sub(r'[\.\"]+$', '', next_content)
        next_content = re.sub(r'\"', '', next_content)
        
        if next_content.startswith('“') and next_content.endswith('”'):
            next_content = next_content[1:-1]
            
        if not re.search(r'<query>.*</query>', next_content):
            next_content = f'<query>{next_content}</query>'
            
        text = text.replace(next_match.group(1), next_content)

    answer_matches = re.finditer(r'So the answer is (.*?)(?= So the answer is|$)', text, re.DOTALL)
    for match in answer_matches:
        answer_content = match.group(1).strip()
        answer_content = re.sub(r'[\.\"]+$', '', answer_content)
        if answer_content.startswith('“') and answer_content.endswith('”'):
            answer_content = answer_content[1:-1]
            
        if not re.search(r'<answer>.*</answer>', answer_content):
            answer_content = f'<answer>{answer_content}</answer>'
            
        text = text.replace(match.group(1), answer_content)

    return text

class ReasonRAGPipeline(BasicPipeline):
    BEGIN_REASONING_PROMPT = """You are an assistant for question answering with access to a retrieval tool. Upon receiving a question, your task is to:
* Analyze and Decompose the Question: Break the question into smaller, manageable sub-questions to ensure all aspects are addressed.
* Evaluate Your Knowledge: Assess each sub-question or component:
- Identify parts you can confidently answer based on your existing knowledge.
- Pinpoint parts that require additional information or verification through retrieval tools.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Respond Format:
If your knowledge is sufficient to answer the question, conclude with:
"So the answer is <answer>{answer_format}</answer>"
If retrieval is necessary to provide a complete answer, conclude with:
"So the next query is <query>query</query>"
"""

    DOCUMENT_ANALYSIS_PROMPT = """You are an information retrieval assistant. Your task is to extract relevant evidence from the provided Wikipedia documents based on the latest query.

    Instructions:

    * Identify key terms or concepts in the query.
    * Search the documents for evidence that supports the query.
    * Response format:
    If relevant evidence is found, output:
       Based on the query, the relevant evidence is <evidence>evidence</evidence>.
    If no relevant evidence is found, output:
       <evidence>None</evidence>.
"""

    REASONING_PROMPT = """You are a question-answering assistant with access to a retrieval tool. Your goal is to provide a concise and accurate reasoning process.
Instructions:
* Error Reflection: If errors exist in previous thoughts, identify and correct them. Skip this step if no errors are present.
* Information Sufficiency: Evaluate whether the current information is sufficient to fully and accurately answer the question. If additional retrieval is needed, deconstruct the question and generate the next query. Avoid repeating previous queries. If no meaningful new query can be generated, explain why and provide an answer based on the current information.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Conclusion:
If generating an answer:
"So the answer is <answer>{answer_format}</answer>".
If more retrieval is needed:
"So the next query is <query>query</query>".
    """

    ANSWER_GENERATION_PROMPT = """You are a reasoning assistant with retrieval. Give a precise and very concise final answer for the given question, conclude with 'So the answer is <answer>{answer_format}</answer>'. Keep your final answer brief and to the point, followed without any explanation.
"""

    EVALUATION_PROMPT = """An agent is tasked with answering a question using a retrieval tool. 
    Critically assess its intermediate reasoning process to determine if it leads to the correct answer. 
    Identify all flaws, inconsistencies, and mistakes in the thought process. 
    Every imperfection, no matter how small, must be acknowledged. 
    Evaluate how effectively the reasoning supports the final answer and the overall accuracy of the response. 
    Ensure the evaluation is extremely harsh, leaving no leniency. 
    Even if the answer seems close to correct, do not award full marks to maintain strict grading standards. 
    Assign a score between [0, 100] based on the severity of flaws and the reasoning’s accuracy in leading to the golden answer.
Respond briefly and conclude with: So the score is [Score].
"""

    def __init__(self, config, prompt_template=None, answer_format="answer", retriever=None, generator=None, max_iter=8, max_children=3,
                 max_rollouts: int = 64, c: float = 1.414, default_uct_score: float = float("inf"), beta=0.95,
                 batch_size=50, max_workers=50):
        self.begin_reasoning_prompt = PromptTemplate(
            config=config,
            system_prompt=f"{self.BEGIN_REASONING_PROMPT.format(answer_format=answer_format)}",
            user_prompt="Question: {question}",
            enable_chat=True
        )

        self.document_analysis_prompt = PromptTemplate(
            config=config,
            system_prompt=f"{self.DOCUMENT_ANALYSIS_PROMPT}",
            user_prompt="Question: {question}. Reference: <reference>{reference}</reference>",
            reference_template="Wikipedia Title: {title}\n{text}\n\n",
            enable_chat=True
        )

        self.reasoning_prompt = PromptTemplate(
            config=config,
            system_prompt=f"{self.REASONING_PROMPT.format(answer_format=answer_format)}",
            user_prompt="Question: {question}",
            enable_chat=True
        )

        self.evaluation_prompt = PromptTemplate(
            config=config,
            system_prompt=f"{self.EVALUATION_PROMPT}",
            user_prompt="Question: {question}",
            enable_chat=True
        )

        self.answer_generation_prompt = PromptTemplate(
            config=config,
            system_prompt=f"{self.ANSWER_GENERATION_PROMPT.format(answer_format=answer_format)}",
            user_prompt="Question: {question}",
            enable_chat=True
        )

        super().__init__(config, self.begin_reasoning_prompt)
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
        self.max_iter = max_iter
        self.max_children = max_children
        self.max_rollouts = max_rollouts
        self.c = c
        self.default_uct_score = default_uct_score
        self.index = 0
        self.beta = beta

        self.batch_size = batch_size
        self.max_workers = max_workers or os.cpu_count()

        self.stop_tokens = ["<|im_end|>", "<|endoftext|>", "</answer>", "</query>", "</evidence>"]
        
    def initialize(self, item) -> MCTSRAGNode:
        self.index = 0
        thoughts = []
        question = item.question
        iter_num = 0
        next_state = "begin_reasoning"
        node = MCTSRAGNode(self.index, question, None, -1, step=iter_num,
                           next_state=next_state, question=question,
                           thoughts=thoughts, golden_answers=item.golden_answers)
        return node

    def search(self, item):
        root = self.initialize(item)
        for _ in range(self.max_rollouts):
            leaf = self.select(root)
            child = self.expand(leaf)
            Q, reward = self.simulate(child)
            self.backpropagate(child, Q, reward)

        self.tree2log(root, item)

        return root

    def tree2log(self, root, item):
        q = queue.Queue()
        q.put(root)

        while not q.empty():
            node = q.get()
            self.node2log(node, item)

            for child in node.children:
                q.put(child)

    def node2log(self, node, item):
        node_dict = node.node_dict
        node_dict["parent_id"] = node.parent_id
        node_dict["children_ids"] = node.children_ids

        item.update_output(
            f"intermediate_node_{node.id}",
            node_dict
        )

    def select(self, node: MCTSRAGNode) -> MCTSRAGNode:
        while not self.is_terminal(node):
            if not node.is_fully_expanded(self.max_children):
                return node
            node = self._select_child(node)
        return node

    def expand(self, node: MCTSRAGNode) -> MCTSRAGNode:
        if node.is_fully_expanded(self.max_children):
            return node

        child = self.get_next_state(node, True)
        return child

    def _select_child(self, node: MCTSRAGNode) -> MCTSRAGNode:
        return max(node.children, key=lambda child: self._uct(child))

    def _uct(self, node: MCTSRAGNode) -> float:
        if node.N == 0:
            return self.default_uct_score
        return node.Q + self.c * math.sqrt(math.log(node.parent.N) / node.N)

    def _best_child(self, node: MCTSRAGNode) -> MCTSRAGNode:
        return max(node.children, key=lambda child: child.N)

    def simulate(self, node):
        Q = self.evaluate_thoughts(node)
        reward = self.get_reward(node) if self.is_terminal(node) else None
        return Q, reward

    def is_terminal(self, node: MCTSRAGNode) -> bool:
        if node.next_state is None or node.max_tokens_reached:
            return True

        return False

    def _simulate_policy(self, node: MCTSRAGNode):
        action = node.next_state
        return action

    def get_next_state(self, parent_node: MCTSRAGNode, use_index=False):
        action = parent_node.next_state
        if action is None:
            return None

        response, node_dict = "", {}
        thoughts = copy.copy(parent_node.thoughts)
        if action == "begin_reasoning":
            response, node_dict = self.handle_begin_reasoning(parent_node.question, thoughts)
        elif action == "document_analysis":
            response, node_dict = self.handle_document_analysis(parent_node.question, thoughts)
        elif action == "reasoning":
            response, node_dict = self.handle_reasoning(parent_node.question, thoughts)
        elif action == "answer_generation":
            response, node_dict = self.handle_answer_generation(parent_node.question, thoughts)

        new_child_node_id = -1
        if use_index:
            self.index += 1
            new_child_node_id = self.index

        node_dict["id"] = new_child_node_id
        node_dict["parent_id"] = parent_node.id
        next_action_state_for_child = self.next_state(action, response, parent_node.step)

        child_node = MCTSRAGNode(
            new_child_node_id,
            parent_node.S + " " + response,
            parent_node,
            parent_node.id,
            parent_node.step + 1,
            next_action_state_for_child,
            parent_node.question,
            thoughts,
            parent_node.golden_answers,
            node_dict
        )

        if self.prompt_template.check_prompt_length(node_dict["input_prompt"]):
            child_node.max_tokens_reached = True

        if use_index:
            parent_node.add_child(child_node)

        return child_node

    def get_reward(self, node: MCTSRAGNode):
        pred = extract_answer(node.node_dict["response"])
        golden_answers = node.golden_answers
        evaluator = F1_Score(self.config)
        reward = evaluator.token_level_scores(pred, golden_answers)
        reward = reward['f1'] * self.beta ** node.step
        return reward

    def backpropagate(self, node, Q, reward):
        while node:
            node.update_Q(Q)
            if reward is not None:
                node.update_reward(reward)
            node = node.parent
            
    def next_state(self, current_action: str, response: str, iter_num: int):
        action_transitions = {
            "begin_reasoning": lambda res: None if re.search(r'<answer>.*?</answer>',
                                                                            res, re.DOTALL) else "document_analysis",
            "reasoning": lambda res: None if re.search(r'<answer>.*?</answer>',
                                                                            res, re.DOTALL) else "document_analysis",
            "document_analysis": lambda res: "reasoning",
            "answer_generation": lambda res: None,
        }

        # Retrieve the function for the current action
        transition_function = action_transitions.get(current_action)

        if iter_num == self.max_iter - 1:
            return "answer_generation"

        if iter_num < self.max_iter-1 and transition_function:
            return transition_function(response)

        return None

    def extract_answer(self, response: str) -> str:
        match = re.search(r'So the answer is\s*(.*?)(?=\n|$)', response, re.IGNORECASE | re.DOTALL)
        if not match:
            return ""

        text = match.group(1).strip()
        # Remove special tokens
        text = re.sub(r'</?(answer|query|evidence)>', '', text)
        return text.strip()

    def extract_query(self, response: str) -> str:
        match = re.search(r'So the next query is\s*(.*?)(?=\n|$)', response, re.IGNORECASE | re.DOTALL)
        if not match:
            return ""

        text = match.group(1).strip()
        # Remove special tokens
        text = re.sub(r'</?(answer|query|evidence)>', '', text)
        return text.strip()

    def delete_tokens(self, response: str) -> str:
        cleaned_response = re.sub(r'<answer>.*?</answer>|<query>.*?</query>', '', response, flags=re.DOTALL)
        return cleaned_response

    def initialize(self, item) -> MCTSRAGNode:
        self.index = 0
        thoughts = []
        question = item.question
        iter_num = 0
        next_state = "begin_reasoning"
        node = MCTSRAGNode(self.index, question, None, -1, step=iter_num,
                           next_state=next_state, question=question,
                           thoughts=thoughts, golden_answers=item.golden_answers)
        return node

    def test_item(self, item):
        root = self.initialize(item)
        node = root

        while not self.is_terminal(node):
            next_node = self.get_next_state(node, use_index=True)
            node = next_node

        self.tree2log(root, item)
        if "answer" in node.node_dict and node.node_dict["answer"] is not None:
            item.update_output("pred", node.node_dict["answer"])
        else:
            item.update_output("pred", "none")
        return node

    def process_annotation(self, dataset, do_eval=True, pred_process_fun=extract_answer):
        for item in tqdm(dataset, desc="Inference: "):
            self.search(item)

        save_data(self.config["save_dir"], dataset)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, batch_size=16, pred_process_fun=None):
        all_dataset_list = []
        for batch_dataset in tqdm(get_batch_dataset(dataset, batch_size=batch_size), desc="Batch dataset: "):
            batch_dataset = self.run_batch(batch_dataset)
            all_dataset_list.append(batch_dataset)
        dataset = merge_batch_dataset(all_dataset_list)
        save_data(self.config["save_dir"], dataset)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def get_flags(self, responses):
        flags = []
        for response in responses:
            if "So the next query is" in response:
                flag = "query"
            elif "So the answer is" in response:
                flag = "answer"
            elif "<evidence>" in response:
                flag = "evidence"
            else:
                flag = "None"
            flags.append(flag)
        return flags

    def get_querys(self, responses):
        querys = []
        for response in responses:
            querys.append(self.extract_query(response))
        return querys

    def get_answers(self, responses):
        answers = []
        for response in responses:
            answers.append(self.extract_answer(response))
        return answers

    def all_finished(self, next_flags):
        return all(flag in ["answer", "finish"] for flag in next_flags)

    def run_batch(self, dataset):
        for item in dataset:
            item.update_output('finish_flag', False)
            item.update_output('iteration_count', 0)
            item.update_output('previous_thoughts', [])
            item.update_output('flag', None)
            item.update_output('query', None)
            item.update_output('answer', None)

        input_prompts = [self.begin_reasoning_prompt.get_string(question=item.question) for item in dataset]
        responses = self.generator.generate(input_prompts, stop=self.stop_tokens)

        for i, item in enumerate(dataset):
            response = responses[i]
            item.previous_thoughts.append(response)
            item.flag = self.get_flags([response])[0]
            item.query = self.get_querys([response])[0]
            item.answer = self.get_answers([response])[0]

            node_dict = {
                "action_name": "begin_reasoning",
                "input_prompt": input_prompts[i],
                "response": response,
                "query": item.query,
                "answer": item.answer,
            }
            item.update_output(f"intermediate_node_0", node_dict)
            if item.flag in ["finish", "answer"]:
                item.finish_flag = True

        for step in range(self.max_iter):
            exist_items = [item for item in dataset if not item.finish_flag]
            if not exist_items:
                break

            active_questions = [item.question for item in exist_items]
            active_previous_thoughts = [item.previous_thoughts for item in exist_items]
            active_flags = [item.flag for item in exist_items]
            active_querys = [item.query for item in exist_items]

            question_thoughts_list = [
                q + "\nPrevious Thoughts: " + " ".join(thoughts)
                for q, thoughts in zip(active_questions, active_previous_thoughts)
            ]

            retrieval_results = self.retriever.batch_search(active_querys)

            input_prompts = []
            for i, item in enumerate(exist_items):
                if item.iteration_count >= self.max_iter - 1:
                    input_prompts.append(self.answer_generation_prompt.get_string(question=question_thoughts_list[i]))
                elif "query" in item.flag:
                    input_prompts.append(self.document_analysis_prompt.get_string(
                        question=question_thoughts_list[i], retrieval_result=retrieval_results[i]
                    ))
                elif "evidence" in item.flag:
                    input_prompts.append(self.reasoning_prompt.get_string(question=question_thoughts_list[i]))
                else:
                    input_prompts.append(self.answer_generation_prompt.get_string(question=question_thoughts_list[i]))

            responses = self.generator.generate(input_prompts, stop=self.stop_tokens)
            for i, item in enumerate(exist_items):
                response = responses[i]
                item.previous_thoughts.append(response)
                item.iteration_count += 1
                item.flag = self.get_flags([response])[0]
                item.query = self.get_querys([response])[0]
                item.answer = self.get_answers([response])[0]

                node_dict = {
                    "action_name": "unknown",
                    "input_prompt": input_prompts[i],
                    "response": response,
                    "query": item.query,
                    "answer": item.answer,
                }
                if "evidence" in item.flag:
                    node_dict["action_name"] = "document_analysis"
                    node_dict["retrieval_result"] = retrieval_results[i]
                elif "query" in item.flag:
                    node_dict["action_name"] = "query_generation"
                elif "answer" in item.flag:
                    node_dict["action_name"] = "answer_generation"
                elif "finish" in item.flag:
                    node_dict["action_name"] = "finish"

                item.update_output(f"intermediate_node_{item.iteration_count}", node_dict)

                if item.flag in ["finish", "answer"] or item.iteration_count >= self.max_iter:
                    item.finish_flag = True

        for i in range(len(dataset)):
            dataset[i].pred = dataset[i].answer

        return dataset

    def handle_begin_reasoning(self, question, thoughts):
        begin_reasoning = [self.begin_reasoning_prompt.get_string(question=question)]
        begin_response = self.generator.generate(begin_reasoning)[0]
        begin_response = process_text(begin_response)
        thoughts.append(begin_response)

        query_matches = re.findall(r'<query>(.*?)</query>', begin_response)
        extracted_query = self.delete_tokens(query_matches[-1]) if query_matches else None

        answer_matches = re.findall(r'<answer>(.*?)</answer>', begin_response)
        extracted_answer = self.delete_tokens(answer_matches[-1]) if answer_matches else None

        node_dict = {
            "action_name": "begin_reasoning",
            "input_prompt": begin_reasoning[0],
            "response": begin_response,
            "next_query": extracted_query,
            "answer": extracted_answer,
        }

        return begin_response, node_dict

    def handle_reasoning(self, question, thoughts):
        question_thoughts = question + "\nPrevious Thoughts: " + " ".join(thoughts)
        reasoning_prompt = [self.reasoning_prompt.get_string(question=question_thoughts)]
        response = self.generator.generate(reasoning_prompt)[0]
        response = process_text(response)
        thoughts.append(response)

        query_matches = re.findall(r'<query>(.*?)</query>', response)
        extracted_query = self.delete_tokens(query_matches[-1]) if query_matches else None

        answer_matches = re.findall(r'<answer>(.*?)</answer>', response)
        extracted_answer = self.delete_tokens(answer_matches[-1]) if answer_matches else None

        node_dict = {
            "action_name": "reasoning",
            "input_prompt": reasoning_prompt[0],
            "response": response,
            "next_query": extracted_query,
            "answer": extracted_answer,
        }

        return response, node_dict

    def handle_document_analysis(self, question, thoughts):
        extracted_query = self.delete_tokens(self.extract_query(thoughts[-1]))
        id2doc = {}
        doc2score = {}
        retrieval_result, scores = self.retriever.search(extracted_query, return_score=True)
        for doc_item, score in zip(retrieval_result, scores):
            id2doc[doc_item["id"]] = doc_item
            doc_id = doc_item["id"]
            if doc_id in doc2score:
                doc2score[doc_id] = max(doc2score[doc_id], score)
            else:
                doc2score[doc_id] = score

        sorted_doc_score = sorted(doc2score.items(), key=lambda x: x[1], reverse=False)
        sorted_doc_id = [t[0] for t in sorted_doc_score]
        retrieval_result = [id2doc[id] for id in sorted_doc_id]

        question_thoughts = question + "\nPrevious Thoughts: " + " ".join(thoughts)
        grounding_prompt = [self.document_analysis_prompt.get_string(
            question=question_thoughts, retrieval_result=retrieval_result
        )]
        response = self.generator.generate(grounding_prompt)[0]
        thoughts.append(response)

        node_dict = {
            "action_name": "document_analysis",
            "input_prompt": grounding_prompt[0],
            "query": extracted_query,
            "response": response,
            "retrieval_result": retrieval_result,
        }

        return response, node_dict

    def handle_answer_generation(self, question, thoughts):
        question_thoughts = question + "\nPrevious Thoughts: " + " ".join(thoughts)
        answer_generation = [self.answer_generation_prompt.get_string(question=question_thoughts)]
        answer = self.generator.generate(answer_generation)[0]
        answer = process_text(answer)
        thoughts.append(answer)
        pred = extract_answer(answer)
        node_dict = {
            "action_name": "answer_generation",
            "input_prompt": answer_generation[0],
            "response": answer,
            "pred": pred,
        }

        return answer, node_dict

    def evaluate_thoughts(self, node):
        question_thoughts = node.question + "\nGolden Answer: " + " or ".join(
            node.golden_answers) + "\nAgent Reasoning Process: " + " ".join(node.thoughts)
        evaluation_prompt = [self.evaluation_prompt.get_string(question=question_thoughts)]
        evaluation_response = self.generator.generate(evaluation_prompt)[0]
        Q = extract_last_number(evaluation_response)
        Q = float(Q) / 100
        return Q
