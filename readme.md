## introduction to text classification 
#### materials for may/2019 monthly DS meeting 
###### timothy leffel, [`http://lefft.xyz`](http://lefft.xyz)

<br>

### get started 

- see [`norc_text_clf_slides.html`](slides/norc_text_clf_slides.html) for presentation slides and code demos in action. 
- all demo code and datasets referenced in the slides can be found in this repository. 



### repository contents 

```
# $ tree -U -F -L 2 -I 'venv|__pycache__|_ignore|stream_of*|notes_to*|libs|*_slides_files' 

text_classification
│
├── readme.md
│
├── slides/
│   ├── norc_text_clf_slides.html
│   └── norc_text_clf_slides.rmd
│
├── code/
│   ├── classify_utterance_type.py
│   └── classify_tweet_relevance.py
│
├── data/
│   ├── tweet_samples-600.txt
│   └── cmdc_lines-annotated-551.txt
│
└── output/
    ├── cmdc_lines-model_info.txt
    └── tweet_samples-model_info.txt

4 directories, 9 files
```

