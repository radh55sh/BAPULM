import torch
import torch.nn as nn

class BAPULM(nn.Module):
    def __init__(self):
        super(BAPULM, self).__init__()
        self.prot_linear = nn.Linear(1024, 512)
        self.mol_linear = nn.Linear(768, 512)
        self.norm = nn.BatchNorm1d(1024, eps=0.001, momentum=0.1, affine=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(1024, 768)
        self.linear2 = nn.Linear(768, 512)
        self.linear3 = nn.Linear(512, 32)
        self.final_linear = nn.Linear(32, 1)

    def forward(self, prot, mol):
        prot_output = torch.relu(self.prot_linear(prot))
        mol_output = torch.relu(self.mol_linear(mol))
        combined_output = torch.cat((prot_output, mol_output), dim=1)
        combined_output = self.norm(combined_output)
        combined_output = self.dropout(combined_output)
        x = torch.relu(self.linear1(combined_output))
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = torch.relu(self.linear3(x))
        output = self.final_linear(x)
        return output
