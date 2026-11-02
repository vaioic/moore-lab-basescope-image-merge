# OIC-244

This project is to merge two histological images into a single image. With Yue Ma (Moore Lab).

## Getting started

### Prerequisites

- [Python](https://www.python.org/downloads/) version 3.14.0

### Installation

1. Download or clone the GitHub repository
   ```bash
   git clone git@github.com:vaioic/OIC-244.git
   cd OIC-244
   ```

2. Create a python virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment
   ```bash
   .\venv\Scripts\activate
   ```

4. Install the dependencies using Pip
   ```bash
   python -m pip install -r .\requirements.txt
   ```

### Running the code

1. Start the virtual environment if not already loaded
   ```bash
   .\venv\Scripts\activate
   ```

2. Run the script
   ```bash
   python -m merge_data
   ```

A combined TIFF-file will be created in the current directory.

## Issues

If you encounter any issues with running the code or have any questions, please create an [Issue](https://github.com/vaioic/OIC-244/issues) or send an email to opticalimaging@vai.org. If you are reporting a programmatic bug, please include any error messages to aid with troubleshooting.

## Acknowledgements

### Dependencies

This project relies on the following packages:

- pandas v2.3.3
- pybioimageutils v0.0.1

**Note:** For full dependency list, see [requirements.txt](requirements.txt).