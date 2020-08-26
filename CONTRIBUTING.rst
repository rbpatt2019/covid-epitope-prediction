.. _contributing:

For contributors
================

Comments, criticisms, and concerns are always welcome! If you would like to help with development, please follow the steps below.
This project depends on `Poetry <https://poetry.eustace.io>`_ for all things dependency and development related. Make sure it's installed, or else all this will fail. It's an awesome tool, I highly recommend you check it out!

Clone the repo 
--------------

.. code:: sh

    git clone https://github.com/rbpatt2019/covid-epitope-prediction.git
    cd covid-epitope-prediction

Make a new environment
----------------------

Follow your own protocol! I use pyenv for all my env/venv control, so I would do:

.. code:: sh

    pyenv virtualenv covid-epitope-prediction
    pyenv local covid-epitope-prediction

Regardless of how you do it, run the following once its created:

.. code:: sh

    make develop

Start developing
----------------

Checkout the `Makefile <https://github.com/rbpatt2019/covid-epitope-prediction/blob/master/Makefile>`_ for lots of useful commands for testing, linting, and many others! Before committing any changes, I'd strongly recommend creating a new branch:

.. code:: sh

    git checkout -b new_feature

And contribute!
---------------

Once you're ready to share your changes, fork the repository on github. Then, add it as a remote to the repo and push the changes there. 

.. code:: sh

    git remote add origin https://github.com/YOUR_USER/covid-epitope-prediction.git
    git push origin new_feature

Finally, open a pull request, and I'll review it as soon as I can!

If you're a command line nut like me, this can all be done from the command line using `hub <https://github.com/github/hub>`_, a CLI for interacting with the github api. See their `repo <https://github.com/github/hub>`_ for installation instructions. Instead of the above, do:

.. code:: sh

    hub fork --remote-name=origin
    git push origin new_feature
    hub pull-request

This will fork the repo, push your changes, and create a pull request, all without leaving the command line!
