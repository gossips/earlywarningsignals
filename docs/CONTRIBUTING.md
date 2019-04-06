# Contributing to Earylwarningsignals package
First off, thanks for taking the time to contribute!

We've developed a set of guidelines for contributing to our early warning signals toolbox.

## Reporting bugs
When you are creating a bug report, please include as many details as possible.
Things to include:
* description of the issue
* steps to reproduce
* expected behaviour (what you expect to happen)
* actual behaviour (what actually happens)
* reproduces how often (what percentage of the time does it reproduce?)
* versions
* additional information

## Contribution
0. Don't push directly to the original repository
1. Create a fork of the code in your personal _GitHub_ account
2. Edit your code and push it to your forked version
3. Create a pull request

### Tips and tricks
- In order to avoid creating a different fork for each edit, configure your fork to point to the original _upstream_ repository. More information [here](https://help.github.com/en/articles/configuring-a-remote-for-a-fork)
- When solving issues, using commit descriptions as _"solves #2"_ or _"closes #2"_ automatically links the commit with the issue, and closes it.

## Acceptance guidelines
Contributions will be accepted if:
* They improve to code and make sense (explained through comments)
* They don't break the build or unit tests
* They introduce new unit tests
* They highlight and fix typos in any part of the repository
* They don't introduce other dependencies
* They don't interfere with the required input and output of the function, unless they are discussed first.
* They don't mess around with the build/project/tool settings

A few notes about unit test contributions:
* WE USE PYTEST, please do not use anything but that.

A few notes about bugfixes:
* Explain the bug before submitting bugfix pull requests
* A bug fix should also have a unit test dedicated to it.
