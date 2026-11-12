# Contributing

Thanks for taking an interest in this project. It is a student project, so the process is kept simple.

## Reporting issues

Open an issue and describe what happened, what you expected, and how to reproduce it. Include the Python version and the steps you took if you can.

## Making a change

1. Fork the repository.
2. Create a branch for your change: `git checkout -b fix-short-description`.
3. Make the change. Keep it small and focused on one thing.
4. Check that the code still compiles: `python -m py_compile app.py train_model.py`.
5. Open a pull request and explain what you changed and why.

## Notes

- Do not commit dataset files, model binaries, or anything under `data/` other than the README.
- Keep the UI simple and the language plain. If you change the model or the training data, update the results in the README to match.
