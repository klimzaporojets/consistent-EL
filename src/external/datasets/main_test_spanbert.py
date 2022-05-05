import torch

from transformers import AutoModel
from transformers import BertTokenizer

if __name__ == "__main__":
    print("Start")
    model_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/dwie_linker/entity_embeddings/' \
                 'spanbert/spanbert_hf_base'

    model = AutoModel.from_pretrained(model_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tknzr = tokenizer.encode('hi this is me, mr. meeseeks', add_special_tokens=True, max_length=512)
    b = torch.tensor(tknzr).unsqueeze(0)

    out = model(b)

    print('the model has been loaded')
