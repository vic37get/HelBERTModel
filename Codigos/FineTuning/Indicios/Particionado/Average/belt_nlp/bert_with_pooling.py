from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from transformers import BatchEncoding, PreTrainedTokenizerBase

from belt_nlp.bert import BertClassifier
from belt_nlp.splitting import transform_list_of_texts
from tqdm.auto import tqdm
import logging

class BertClassifierWithPooling(BertClassifier):
    """
    The splitting procedure is the following:
        - Tokenize the whole text (if maximal_text_length=None) or truncate to the size maximal_text_length.
        - Split the tokens to chunks of the size chunk_size.
        - Tokens may overlap dependent on the parameter stride.
        - In other words: we get chunks by moving the window of the size chunk_size by the length equal to stride.
        - See the example in https://github.com/google-research/bert/issues/27#issuecomment-435265194.
        - Stride has the analogous meaning here that in convolutional neural networks.
        - The chunk_size is analogous to kernel_size in CNNs.
        - We ignore chunks which are too small - smaller than minimal_chunk_length.

    After getting the tensor of predictions of all chunks we pool them into one prediction.
    Aggregation function is specified by the string parameter pooling_strategy.
    It can be either "mean" or "max".
    """

    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        chunk_size: int,
        stride: int,
        minimal_chunk_length: int,
        pooling_strategy: str = "mean",
        accumulation_steps: int = 1,
        maximal_text_length: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        super().__init__(
            batch_size,
            learning_rate,
            epochs,
            accumulation_steps,
            tokenizer,
            neural_network,
            pretrained_model_name_or_path,
            device,
            many_gpus,
        )

        self.loggerf = logging.getLogger(name='arq')
        self.loggerf.propagate = False
        self.loggerf.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler_file =  logging.FileHandler(filename='BELT-Classifier.log',mode='w')
        handler_file.setFormatter(formatter)
        handler_file.setLevel(logging.DEBUG)
        self.loggerf.addHandler(handler_file)

        self.loggert = logging.getLogger(name='tela')
        self.loggert.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler_screen = logging.StreamHandler()
        handler_screen.setFormatter(formatter)
        handler_screen.setLevel(logging.DEBUG)
        self.loggert.addHandler(handler_screen)        

        self.chunk_size = chunk_size
        self.stride = stride
        self.minimal_chunk_length = minimal_chunk_length
        if pooling_strategy in ["mean", "max"]:
            self.pooling_strategy = pooling_strategy
        else:
            raise ValueError("Unknown pooling strategy!")
        self.maximal_text_length = maximal_text_length

        additional_params = {
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "minimal_chunk_length": self.minimal_chunk_length,
            "pooling_strategy": self.pooling_strategy,
            "maximal_text_length": self.maximal_text_length,
        }
        self._params.update(additional_params)

        self.device = device
        self.collate_fn = BertClassifierWithPooling.collate_fn_pooled_tokens

    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        """
        Transforms list of N texts to the BatchEncoding, that is the dictionary with the following keys:
            - input_ids - List of N tensors of the size K(i) x 512 of token ids.
                K(i) is the number of chunks of the text i.
                Each element of the list is stacked Tensor for encoding of each chunk.
                Values of the tensor are integers.
            - attention_mask - List of N tensors of the size K(i) x 512 of attention masks.
                K(i) is the number of chunks of the text i.
                Each element of the list is stacked Tensor for encoding of each chunk.
                Values of the tensor are booleans.

        These lists of tensors cannot be stacked into one tensor,
        because each text can be divided into different number of chunks.
        """
        tokens = transform_list_of_texts(
            texts, self.tokenizer, self.chunk_size, self.stride, self.minimal_chunk_length, self.maximal_text_length, self.loggerf
        )
        return tokens

    def _evaluate_single_batch(self, batch: tuple[Tensor],optimizer) -> Tensor:
        input_ids = batch[0]
        #self.loggerf.debug("[evaluate_single_batch] input_ids shape: %s | inputs ids element shape: %s ", len(input_ids),input_ids[0].shape)
        attention_mask = batch[1]
        #self.loggerf.debug("[evaluate_single_batch] attention_mask: %s | attention_mask element shape: %s ", len(attention_mask),attention_mask[0].shape)
        number_of_chunks = [len(x) for x in input_ids]
        #self.loggerf.debug("[evaluate_single_batch] number_of_chunks: %s", number_of_chunks)

        # concatenate all input_ids into one batch
        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        #input_ids_combined_tensors = torch.stack([torch.tensor(x).to(self.device) for x in input_ids_combined])

        # concatenate all attention masks into one batch
        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())
        
        #attention_mask_combined_tensors = torch.stack(
        #    [torch.tensor(x).to(self.device) for x in attention_mask_combined]
        #)

        # get model predictions for the combined batch
        #preds = self.neural_network(input_ids_combined_tensors, attention_mask_combined_tensors)

        #preds = preds.flatten().cpu()

        # split result preds into chunks
    
        batch_chunks=2*len(batch[0])
        #self.loggerf.debug("[evaluate_single_batch] batch_chunks: %s" ,batch_chunks)
        preds=Tensor()
        #self.loggerf.debug("[evaluate_single_batch] memory  before for : %s",torch.cuda.memory_allocated())
        loop2 = tqdm(range(0, len(input_ids_combined), batch_chunks), leave=True, colour='yellow')
        for i in loop2:
            input_ids_temp=input_ids_combined[i:i + batch_chunks]
            input_ids_combined_tensors = torch.stack([torch.tensor(x).to(self.device) for x in input_ids_temp])
            #self.loggerf.debug("[evaluate_single_batch] input_ids_temp shape: %s | element: %s", len(input_ids_temp),len(input_ids_temp[0]))
            attention_mask_temp=attention_mask_combined[i:i + batch_chunks]
            attention_mask_combined_tensors = torch.stack([torch.tensor(x).to(self.device) for x in attention_mask_temp])
            #self.loggerf.debug("[evaluate_single_batch] input_ids_combined_tensors shape: %s", input_ids_combined_tensors.shape)
            # get model predictions for the combined batch
            #self.loggerf.debug("[evaluate_single_batch] memory before neural : %s",torch.cuda.memory_allocated())
            preds_chunks = self.neural_network(input_ids_combined_tensors, attention_mask_combined_tensors)
            loop2.set_description(f'Eval Single Batch | Batch: {i}')
            #self.loggerf.debug("[evaluate_single_batch] memory after neural : %s",torch.cuda.memory_allocated())
            preds_chunks = preds_chunks.flatten().cpu()
            #self.loggerf.debug("[evaluate_single_batch] preds chunks: %s | shape: %s",preds_chunks, preds_chunks.shape)
            preds=torch.cat([preds,preds_chunks])
            #self.loggerf.debug("[evaluate_single_batch] memory before zero grad : %s",torch.cuda.memory_allocated())
            #self.loggerf.debug("[evaluate_single_batch] memory before zero grad : %s",torch.cuda.memory_allocated())
        #self.loggerf.debug("[evaluate_single_batch] memory after for : %s",torch.cuda.memory_allocated())

        # split result preds into chunks

        #self.loggerf.debug("[evaluate_single_batch] preds: %s | shape: %s",preds, preds.shape)
        preds_split = preds.split(number_of_chunks)
        
        #self.loggerf.debug("[evaluate_single_batch] pooled preds split: %s | shape: %s",preds_split, preds_split[0].shape)

        # pooling
        if self.pooling_strategy == "mean":
            pooled_preds = torch.cat([torch.mean(x).reshape(1) for x in preds_split])
        elif self.pooling_strategy == "max":
            pooled_preds = torch.cat([torch.max(x).reshape(1) for x in preds_split])
        else:
            raise ValueError("Unknown pooling strategy!")
        #self.loggerf.debug("[evaluate_single_batch] pooled preds: %s | shape: %s",pooled_preds, pooled_preds.shape)
        #self.loggerf.debug("*********[evaluate_single_batch] **************************")

        return pooled_preds

    @staticmethod
    def collate_fn_pooled_tokens(data):
        input_ids = [data[i][0] for i in range(len(data))]
        attention_mask = [data[i][1] for i in range(len(data))]
        if len(data[0]) == 2:
            collated = [input_ids, attention_mask]
            #print("[collate_fn_pooled] collated shape: ", len(collated), len(collated[0]), len(collated[1]))
        else:
            labels = Tensor([data[i][2] for i in range(len(data))])
            collated = [input_ids, attention_mask, labels]
            #print("[collate_fn_pooled] collated shape: ", len(collated), len(collated[0]), len(collated[1]), collated[2])
        
        return collated
