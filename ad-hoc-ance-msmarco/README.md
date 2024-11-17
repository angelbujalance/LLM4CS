---
license: apache-2.0
library_name: sentence-transformers
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
pipeline_tag: sentence-similarity
---

# sentence-transformers/msmarco-roberta-base-ance-firstp

This is a port of the [ANCE FirstP Model](https://github.com/microsoft/ANCE/) to [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.



## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-ance-firstp')
embeddings = model.encode(sentences)
print(embeddings)
```



## Evaluation Results



For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name=sentence-transformers/msmarco-roberta-base-ance-firstp)



## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  (2): Dense({'in_features': 768, 'out_features': 768, 'bias': True, 'activation_function': 'torch.nn.modules.linear.Identity'})
  (3): LayerNorm(
    (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
)
```

## Citing & Authors

Have a look at: [ANCE Model](https://github.com/microsoft/ANCE/)