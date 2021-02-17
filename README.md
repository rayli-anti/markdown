# GLaDOS Take Home Challenge: *Tagging*

Our main vision within the GLaDOS Machine Learning Engineering team at Sky is to create a framework and tooling for Data Scientists to work effectively from experimentation to production. We want to put ML models into production, *and keep them there*!

At Sky, every piece of content (TV Show, Movie, Sports event, etc) has *metadata* attached to it - the title, the release year, synopsis, etc.

A *tag* is a kind of metadata that quickly describes a piece of content. For example, the movie `Die Hard` could have the following tags:

```
heist, christmas, hostages, police, Bruce Willis, Alan Rickman, Los Angeles, skyscraper
```

Imagine a Data Scientist has created a brand new model to automatically create these tags for a given movie or TV programme, based on the synopsis alone. They're excited to put it into production, and need your help to do so.

> n.b. Don't worry about completing every part of this exercise, and don't spend too much time on it! We've intentionally made this open ended, to allow you to show off if you want to, or to prove your skills within a fair amount of time.

## Part 0: Setup
Before you start, make sure you have a python development environment set up. The project assumes Python 3.7. We're currently using [poetry](https://python-poetry.org/) to manage dependencies, which we advise you to do the same. ```poetry``` has many advantages over ```pip``` or ```pipenv``` but if you want to use either of those then you can easiliy make a ```Pipfile``` from the included ```pyproject.toml``` file. 

On `osx` / `linux`, you can just use this command to install poetry:

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```
The installation will modify your Bash profile, but if you are already in a Bash shell, then you will either need to re-run your profile;

```shell
source ~/.bash_profile
```
or just update the ```PATH``` in the current shell directly:
```shell
export PATH="$HOME/.poetry/bin:$PATH"
```
<!-- TODO: set up poetry inside tagging/ -->
Once you've got that installed, run `poetry install` in the root directory of this project, then run `poetry shell`.

Part 1 should be easy to run once you've installed the dependencies. For example, just run:

```
python part1
```

To run the tests (that you should fix and complete), you can just run:

```
pytest part1
```
(both from the root of this directory)

> n.b. The tests will fail when you run them initially, it will be your job to update them (they're not really testing anything at the moment) and make them pass.

## Part 1: Preprocessing
We have some content we'd like to generate tags for. For each piece of content, we have the following metadata (called an `EntityDetails` object):

```
{
  "id": "...",
  "title": "...",
  "shortSynopsis": "...",
  "mediumSynopsis": "...",
  "longSynopsis": "...",
  "genre": "...",
  "subgenres": [ "..." ]
}
```


Before we can use the model to generate the tags, we need to perform some preprocessing to match what the Data Scientist did before training. Sadly they seem to have lost the preprocessing code, but they have remembered the steps they took, so it's up to you to convert these instructions into code.

The model in question takes a text feature as it's input, which is a string containing parts of the `EntityDetails` object.

### *1.1: Creating the Text Feature*
The model requires a text feature, which is a single string comprising all the relevant information from the input EntityDetails. The feature is created as follows:
1. Take the longest of the 3 synopses (even if it is not `longSynopsis`)
2. Prepend the genre to the synopsis, separated by a space
3. Join all the subgenres, separated by spaces, and prepend to the string
containing the subgenre and longest synopsis.

To start, complete the `preprocess` method in [part1/preprocessing_service.py](part1/preprocessing_service.py) so that it will create the text feature from a single EntityDetails object. 

Make sure to add unit tests in [part1/test_preprocessing_service.py](part1/test_preprocessing_service.py). Remember that Pytest will be installed if you used `poetry`, but feel free to use whatever testing framework you like. 

You'll find a some example data that we'd like to create tags for in [example_data/](example_data/README.md).
In `train/` you'll find labelled examples ready to be trained with, and you'll find a smaller set of unlabelled examples in `eval` which are ready for prediction. Note how some of the data is missing or in a bad format. Assume a "~" means n/a or null.

Running `python part1` from the root of this directory will initially loop through the example data, and attempt to preprocess each example using the `preprocess` method you create (see [part1/\_\_main\_\_.py](part1/__main__.py). Note that you are allowed to edit `part1/__main__.py`!


### *1.2: Cleaning the Text Feature*
To make things easier for the model we must remove useless or misleading
information. Once you have the text feature from the previous stage you must
then:
1. Remove all numbers
2. Remove punctuation: `?` `!` `'` `"` `#` `:` 
1. Replace separators and special characters with spaces, for example: `.` `,` `)` `(` `\` `/` `-`
2. Replace sequences of more than one space with one space.
3. Replace newlines with spaces
4. Remove all words shorter than a configurable length
5. Lemmatize all words
6. Remove all stop words

You can use any of the following packages, in combination, or something else!
- [textblob](https://textblob.readthedocs.io/en/dev/quickstart.html#words-inflection-and-lemmatization)
- [Spacy](https://spacy.io/)
- [Keras Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text)

These are specified in `pyproject.toml`, so if you install the dependencies via poetry this will be installed in the virtual environment.

Implement a `clean` method or class which takes in a string and makes the above modifications. Call this from your `preprocess` method in [part1/preprocessing_service.py](part1/preprocessing_service.py), with the text feature you created in the previous part.


### Part 1.3: *Tokenization*
To finish off our preprocessing service, we need to tokenize our text feature and convert to a vector format, in order for the ML model to do some mathematical magic on. The Data Scientist tells you that they used the [Keras Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer), but they've lost the fitted vocabulary they created. They fitted it based on the training data, so you'll need to redo this. 

Inside [part1/\_\_main\_\_.py](part1/__main__.py), create an instance of the Keras Tokenizer, and fit the tokenizer on the [training catalogue](example_data/train/train_catalogue.csv). Save the tokenizer to a JSON file - the Data Scientist forgot to do this! Ensure that your code only fits the tokenizer and creates the JSON file the first time it is run - on subsequent runs, it should load the saved state of the Tokenizer using the JSON file - see the Tensorflow docs on how to do this.

Refactor the [PreprocessingService](part1/preprocessing_service.py) so that it can be initialized with the tokenizer, and then use the tokenizer in `preprocess` to convert the cleaned text feature into a sequence of integers.


Remember to add unit tests!

**The final outcome of [part1](part1/__main__.py) will be the saved JSON vocabulary, and the printed output of preprocessing the eval catalogue to create the sequence of integers that will be used for prediction.**

---

## Part 2: Taking it further

Now we've got preprocessing ready, we'll be able to call the model with our content and generate a bunch of lovely tags. But what next? 

Please answer the following open-ended questions in `part2.txt`:

*2.1:* Eventually your team creates an API that allows users to send in an `EntityDetails` object, process the data, and query the model in order to receive the predicted tags back. How would you ensure that this API copes with thousands of users at once?

*2.2:* You've successfully put the model in production. But six months down the line, the model doesn't seem to be performing very well, and the users are complaining. Why might this be, and what steps would you take to improve the performance?

*2.3:* How would you approach the tagging problem if you were tackling it from scratch?


---
## Part 3: Extras

> n.b. This part is for you to show off, **if and only if you want to**. We're not expecting anyone to do this, but want to give people the option.

Explore the data given to you in [example_data/](example_data/README.md), and see what you can get out of it. This *might* be one of the following, or something else you think would be interesting:
- a notebook on exploratory data analysis
- a wrapper around the preprocessing service that serves it as an API over HTTP
- a simple but powerful model
- a full blown ML pipeline in TFX or anther framework

Put everything you do in this part in a folder title `part3`. Feel free to import code from part 1!

---

## Part 4: Submit
Please zip up the entire directory and send it back to us.

Make sure to include your answers to part 2 in `part2_answers.txt`.

We need to be able to run your code and tests, so please add any instructions if needed, or specify any extra dependencies in `pyproject.toml` by using `poetry add <package>`.


