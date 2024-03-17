# 

Guides on Code Usage
------------------
Commands to train and test retriever models are provided in ```commands``` folder. 

To change train/test setting for the model, modify the shell scipts in ```commands``` folder or configuration files in ```conf``` folder.
2. ```train_dpr.sh```: Train DPR model and generate checkpoint files under ```outputs/checkpoint```.
3. ```generate_embeddings.sh```: Generate encoded embeddings for an input corpus under ```outputs/embeddings```.
4. ```validate_retriever.sh```: Generate scores for an input question set and corpus embeddings under ```outputs/validation```.
5. ```evaluate_dpr.sh```: (Optional) calulates MAP, recall, MRR metrics for generated scores from step 4. Corresponding query set and answer set (qid-pid pairs) must be provided. Can be replaced with ```evaluate_script.ipynb```

Refer to [Github repository for DPR][dpr_github] and [documentation for Hydra][hydra_doc] for more details.

[dpr_paper]: https://arxiv.org/abs/2004.04906
[dpr_github]: https://github.com/facebookresearch/DPR
[hydra_doc]: https://hydra.cc/docs/intro/


Training Sample Format
------------------

<pre>
<code>
[
    {
        "question": ... ,
        "answers": [],
        "positive_ctxs": [
            {
                "title": ... ,
                "text": ... ,
                "passage_id": ...
            }
        ],
        "negative_ctxs": [
            {
                "title": ... ,
                "text": ... ,
                "passage_id": ...
            },
            ...
        ],
        "hard_negative_ctxs": [
            {
                "title": ... ,
                "text": ... ,
                "passage_id": ... 
            },
            ...
        ]
    }
]
</code>
</pre>