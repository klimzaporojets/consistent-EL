### On working notes 
This *working notes* are started on 20/10/2021 in order to give a better follow-up of the
code, improvements and TODOs. This is complementary to the TODOs in ```todos_general.txt```. 
### Code improvement TODOs: 
1. [ ] in ```modules.transformers.WrapperSpanBERT_X``` in ```forward```,    
  not sure what happened, but .last_hidden_state seems no to be found anymore. So commented the
  following line of code: 
  ```outputs = self.spanbert_model(segmented_doc, attention_mask=segments_mask).last_hidden_state```
