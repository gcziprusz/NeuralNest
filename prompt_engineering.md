# prompt engineering techniques
## zero shot prompting
In zero-shot prompting you are prompting the model to see if it can infer the task from the structure of your prompt.
```
Prompt: "Coach B, thank you for the inspiring pep talk before the match!
Sentiment: ?"
```
## few shot prompting
In few-shot prompting, you not only provide the structure to the model, but also two or more examples.
```
Prompt: "Player A, you missed the open net in the final minutes of the game!"
Sentiment: Negative

Prompt: "Our team just won the championship! Celebrating tonight!"
Sentiment: Positive

Prompt: "Coach B, thank you for the inspiring pep talk before the match!"
Sentiment: ?"
```
## specify output format
You can also specify the format in which you want the model to respond.
In the example below, you are asking to "give a one word response".
## role prompting
Roles give context to LLMs what type of answers are desired.
## summarize
Summarizing a large text is another common use case for LLMs
## provide new info , context
A model's knowledge of the world ends at the moment of its training - so it won't know about more recent events. But the user can provide missing info / context.
## chain-of-thought for reasonnig
LLMs can perform better at reasoning and logic problems if you ask them to break the problem down into smaller steps.

Since LLMs predict their answer one token at a time, the best practice is to ask them to think step by step, and then only provide the answer after they have explained their reasoning.


## Model-Graded Evaluation: Summarization
Interestingly, you can ask a LLM to evaluate the responses of other LLMs.

## Safety
- llama guard user(input)
- llama guard agent(output)

# Summary

# Clear Specific instructions
# Iterative prompt development
# Give the LLM time to think

# Capabilities 

### Summarizing
### Inferring Sentiment analysis
### Transforming aka Translating
### Expanding

## Settings 
### Temperature 

## Special usecase
### Custom Chatbots

## Tools 
### redlines for diff