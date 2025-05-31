from models.basemodel import BaseModel
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.torch_lib.tensor_loader import FastTensorDataLoader
from models.basemodule_embedding import EmbeddingWrapper,adjust_input_dim,select_embedding_module
from gpu.gpu_monitor import log_memory_usage

from utils.io_utils import get_output_path

class BaseModelTorch(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        self.device = self.get_device()
        self.gpus = args.gpu_ids if args.use_gpu and torch.cuda.is_available() and args.data_parallel else None
        
        # numerical embedding : 
        # Set num_bins from best parameters if not provided
        # select embedding module, adjust input dimension
        
        #if args.numerical_embedding:
        self._set_num_bins_from_params(params, args)
        self.args.embedding_module = select_embedding_module(self.args)
        self.adjusted_input_dim = adjust_input_dim(self.args)
        # else:
        #     self.adjusted_input_dim = self.args.num_features
    
    # numberical embedding : initialize subclasses then build embedding layer
    def __init_subclass__(cls):
        def init_with_embedding(self, params, args, init=cls.__init__):
            init(self, params, args)
            if args.numerical_embedding and args.model_name != 'transformer':
                self.model = EmbeddingWrapper(self.model, args)
            self.to_device()

        init_with_embedding.__qualname__ = cls.__name__ + ".__init__"
        cls.__init__ = init_with_embedding
        # cls.__init__ will be called after __init_subclass__ is called

    def to_device(self):
        if self.args.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        print(f"On Device:", self.device)
        self.model.to(self.device)

    def get_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.data_parallel:
                device = "cuda"  # + ''.join(str(i) + ',' for i in self.args.gpu_ids)[:-1]
            else:
                device = 'cuda'
        else:
            device = 'cpu'

        return torch.device(device)

    def fit(self, X, y, X_val=None, y_val=None):
        # print("\nInitial GPU memory state:")
        # log_memory_usage()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"])

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float() 

        y = torch.tensor(y)
        y_val = torch.tensor(y_val) 

        # print("\nGPU memory after data tensors creation:")
        # log_memory_usage()

        if self.args.objective == "regression":
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()
 
        train_loader = None
        if self.args.data_loader == "torch":
            train_dataset = TensorDataset(X, y)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=self.args.shuffle,
                                  num_workers=4)
        else:
            train_loader = FastTensorDataLoader(X, y, batch_size=self.args.batch_size, shuffle=self.args.shuffle)

        val_loader = None
        if self.args.data_loader == "torch":
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.val_batch_size, shuffle=self.args.shuffle)
        else:
            val_loader = FastTensorDataLoader(X_val, y_val, batch_size=self.args.val_batch_size, shuffle=self.args.shuffle)

        # print("\nGPU memory after dataloaders creation:")
        # log_memory_usage()

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            # print(f"\nEpoch {epoch} - Starting GPU memory:")
            # log_memory_usage()
            
            for i, (batch_X, batch_y) in enumerate(train_loader):
                # if i == 0:  # Only log first batch to avoid spam
                #     print(f"\nEpoch {epoch} - First batch GPU memory:")
                #     log_memory_usage()
                    
                out = self.model(batch_X.to(self.device))

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if i == 0:  # Only log first batch to avoid spam
                #     print(f"\nEpoch {epoch} - After first batch backward pass:")
                #     log_memory_usage()

            # Early Stopping
            # print(f"\nEpoch {epoch} - Before validation:")
            # log_memory_usage()
            
            val_loss = 0.0
            val_dim = 0
            with torch.no_grad():
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    # try:
                    #     # print(f"size of val batch {val_i}: {batch_val_X.shape}")
                    #     # log_memory_usage()
                    #     out = self.model(batch_val_X.to(self.device))
                    # except RuntimeError as e:
                    #     # print(f"\nError during validation 1 - Current GPU memory:")
                    #     # log_memory_usage()
                    #     raise e

                    out = self.model(batch_val_X.to(self.device))
                    if self.args.objective == "regression" or self.args.objective == "binary":
                            out = out.squeeze()

                    val_loss += loss_func(out, batch_val_y.to(self.device))
                    val_dim += 1


            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            # print(f"\nEpoch {epoch} - After validation:")
            # log_memory_usage()

            print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            # save checkpoints
            # ......

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory="tmp")

            if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                print("Early stopping applies.")
                break

        # Load best model
        self.load_model(filename_extension="best", directory="tmp")
        
        # print("\nFinal GPU memory state:")
        # log_memory_usage()
        
        return loss_history, val_loss_history

    def predict(self, X):
        if self.args.objective == "regression":
            self.predictions = self.predict_helper(X)
        else:
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_helper(X)

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        if self.args.data_loader == "torch":
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                 num_workers=2)
        else:
            test_loader = FastTensorDataLoader(X, batch_size=self.args.val_batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.model(batch_X[0].to(self.device))

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)
                elif self.args.objective == "classification":
                    preds = nn.Softmax(dim=1)(preds)

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    def save_model_and_predictions(self, y_true: np.ndarray, filename_extension=""):
        super().save_model_and_predictions(y_true, filename_extension)
        if self.args.numerical_embedding:
            if self.args.model_name != 'transformer':
                self.record_embedding_parameters(filename_extension)
            else:
                self.record_tokenizer_parameters(filename_extension, directory="transformer_parameters")

    def save_embedded_features(self, X, y, filename_extension="",directory="embedded_features"):
        self.model.eval()

        X = torch.tensor(X).float()  
        bs = self.args.batch_size * 100
        data_loader = None
        if self.args.data_loader == "torch":
            dataset = TensorDataset(X)
            data_loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=False,num_workers=4)
        else:
            data_loader = FastTensorDataLoader(X, batch_size=bs, shuffle=False)

        # for each batch, embed using model.num_emb, and save to to embedded_X
        embedded_X = torch.zeros((X.shape[0], self.adjusted_input_dim))
        with torch.no_grad():
            for i, batch_X in enumerate(data_loader):
                embedded_X[i * bs:(i + 1) * bs] = self.model.num_emb(batch_X[0].to(self.device))
        embedded_X = embedded_X.cpu().numpy()

        filename = get_output_path(self.args, directory=directory, filename="embedded_X", extension=filename_extension, file_type="npy")
        print("Saving embedded features to: ", filename)
        np.save(filename, embedded_X)
        # save rounded version as csv
        np.savetxt(filename.replace(".npy", ".csv"),embedded_X,  fmt='%.2f', delimiter=",")

    def record_embedding_parameters(self, filename_extension="", directory="embedding_parameters"):       
        d = self.model.num_emb.state_dict()
        for key, value in d.items():
            filename = get_output_path(self.args, directory=directory , filename=key, extension=filename_extension,file_type="npy")
            if key == 'M':
                value = value.flatten(start_dim=-2).cpu().numpy()
            else:
                value = value.cpu().numpy()
            np.save(filename, value)
            np.savetxt(filename.replace(".npy", ".csv"), value, fmt='%.2f', delimiter=",")

    def record_tokenizer_parameters(self, filename_extension="", directory="transformer_parameters"):
        d = self.model.tokenizer.state_dict()
        for key, value in d.items():
            key = key.split(".")[-1]
            filename = get_output_path(self.args, directory=directory , filename=key, extension=filename_extension,file_type="npy")
            if key == 'M':
                value = value.flatten(start_dim=-2).cpu().numpy()
            else:
                value = value.cpu().numpy()
            np.save(filename, value)
            np.savetxt(filename.replace(".npy", ".csv"), value, fmt='%.2f', delimiter=",")

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        if filename_extension != "best":
            print("Saving model to: ", filename)
        torch.save(self.model.state_dict(), filename)
    
    def load_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        raise NotImplementedError("This method has to be implemented by the sub class")

    def _set_num_bins_from_params(self, params, args):
        """Set num_bins from best parameters if not provided.
        
        Args:
            params: Dictionary containing model parameters
            args: Arguments object to update with num_bins
        """
        if args.num_bins_source == "":
            if args.num_bins == 0:
                raise ValueError("num_bins is not set and no source provided.")
            else:
                print(f'Using specified embedding dimension: {args.num_bins}')
            return 

        if args.source == 'rtdl-modelwise':
            args.num_bins = params['num_bins']
            print(f'Using embedding dimension from model {args.model_name_with_embedding_suffix} source: {args.num_bins}')
        elif args.source == 'rtdl':
            suffix = args.num_bins_source.removeprefix("model")  
            args.num_bins = params['num_bins'][suffix]
            print(f'Using embedding dimension from {suffix} source: {args.num_bins}')
        else:
            raise ValueError("Invalid source provided. Only 'rtdl' and 'rtdl-modelwise' are supported.")
            
        
            

