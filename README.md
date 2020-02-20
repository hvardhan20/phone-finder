# Phone Finder

This project provides a model trainer to predict the location of a phone in a given image. 

## Getting Started

The main runnable files in this project are `train_phone_finder.py` and `find_phone.py`

### Prerequisites

Please try to run this project from the root of the project directory as much as possible.
Command line arguments **CAN BE ABSOLUTE**.
From the root of this project directory structure, run the `setup.py` script as 
```
python setup.py
```
to install all the project dependencies. This script installs Mask R-CNN package, 
which is the backbone of this entire project. Mask R-CNN package is installed directly from Git source
repository as its PyPI version is not updated with the latest changes. Hence it is buggy. 
Besides Mask R-CNN, there are other packages that are required and are installed from the `requirements.txt`

### Initializing

This project is highly configurable via the `./config/params.json` file. You can


```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
