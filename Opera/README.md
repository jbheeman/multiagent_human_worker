How it works: 

We generate a persona for all the users that are in data/train.json using the prompt that is given.
This persona is then compared against the golden persona information that is written in opera. Evaluation right now is done using from  SentenceTransformers to check similarity, but it can probably be further optimized to use an LLM to do the evaluation too?

After evaluation of closeness ,make_reflective_dataset creates feedback that reflection_lm uses to propose better prompts. More specific, contextual feedback should help.

Here a judge LLM is used so that LLM-generated feedback can be tailored to each trajectory


To get data run the Opera/loaddata.py - this will get u data/test.json, train, val. Each one contains all items the user interacted with

to run GEPA on a prompt, run personaAdapter.py

