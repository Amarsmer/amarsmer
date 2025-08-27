#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random

class PioneerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        hidden_size = 1000  # valeur modifiée pour la couche cachée.
        super(PioneerNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Définir les couches
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # Fonction d'activation (tanh comme dans l'implémentation originale)
        self.activation = nn.Tanh()
        
        # Initialisation des poids avec une distribution uniforme [-1, 1]
        # comme dans l'implémentation originale
        nn.init.uniform_(self.hidden.weight, -1.0, 1.0)
        nn.init.uniform_(self.output.weight, -1.0, 1.0)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x):
        # Conversion en tensor PyTorch si nécessaire
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Passage à travers les couches
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x
    
    def run_nn(self, inputs):
        """Méthode compatible avec l'interface originale"""
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32)
            outputs = self.forward(x)
            return outputs.tolist()
    
    def load_weights_from_json(self, json_obj, hidden_size):
        """Charger les poids à partir d'un objet JSON (format original)"""
        # Convertir les poids d'entrée en tenseurs PyTorch
        input_weights = torch.zeros(self.input_size, hidden_size)
        for i in range(self.input_size):
            for j in range(hidden_size):
                input_weights[i][j] = json_obj["input_weights"][i][j]
        
        # Convertir les poids de sortie en tenseurs PyTorch
        output_weights = torch.zeros(hidden_size, self.output_size)
        for i in range(hidden_size):
            for j in range(self.output_size):
                output_weights[i][j] = json_obj["output_weights"][i][j]
        
        # Affecter les poids au modèle
        with torch.no_grad():
            self.hidden.weight.copy_(input_weights.t())  # Transposer car PyTorch utilise (out_features, in_features)
            self.output.weight.copy_(output_weights.t())
            
    def save_weights_to_json(self):
        """Convertir les poids en format JSON compatible avec l'original"""
        # Convertir les poids d'entrée au format original
        input_weights = []
        hidden_weights = self.hidden.weight.detach().t()  # Transposer pour revenir au format original
        for i in range(self.input_size):
            row = []
            for j in range(self.hidden_size):
                row.append(float(hidden_weights[i][j]))
            input_weights.append(row)
        
        # Convertir les poids de sortie au format original
        output_weights = []
        out_weights = self.output.weight.detach().t()  # Transposer pour revenir au format original
        for i in range(self.hidden_size):
            row = []
            for j in range(self.output_size):
                row.append(float(out_weights[i][j]))
            output_weights.append(row)
        
        return {"input_weights": input_weights, "output_weights": output_weights}