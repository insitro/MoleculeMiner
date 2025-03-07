# MoleculeMiner Server

This is the server component of MoleculeMiner. It uses FastAPI and uvicorn to serve the backend, and some simple HTML and JS for the frontend. The server is designed to be run on a local machine or a server, and can be accessed through a web browser once it is running.

<!-- TOC -->
* [MoleculeMiner Server](#moleculeminer-server)
  * [Installation](#installation)
    * [Step 1: Clone the Repository](#step-1-clone-the-repository)
    * [Step 2: Install Dependencies](#step-2-install-dependencies)
    * [Step 3: Start the Server](#step-3-start-the-server)
    * [Step 4: Access the Server](#step-4-access-the-server)
  * [Background & Usage](#background--usage)
    * [Upload Page](#upload-page)
    * [Viewing/Download Page](#viewingdownload-page)
<!-- TOC -->

## Installation

### Step 1: Clone the Repository

```shell
git clone https://github.com/insitro/moleculeminer.git
cd moleculeminer/server
```

### Step 2: Install Dependencies
You can install the dependencies inside the root repo environment if you have created one already called `molminer`.


```shell
# Install molminer dependencies
pip install -r ../molminer/requirements.txt
# Install server dependencies
pip install -r requirements.txt
```

### Step 3: Start the Server

If you want to provide an OpenAI API key at this point so that all users can , you can set it as an environment variable now and it can be used by all users. Otherwise, you can leave it to the user(s) to provide their own API key.
```shell
export OPENAI_API_KEY=<your-api-key>
```

Then you can start the server **(from the root of this repo)** with the command below.:
```shell
export PYTHONPATH="${PYTHONPATH}:$(pwd)/MolScribe:$(pwd)/MolScribe/molscribe:$(pwd)/molminer"

python server/main.py

# Or for development/debug mode
python server/main.py --logmode DEBUG
```

### Step 4: Access the Server

If you are running on your local machine, you can open a web browser and navigate to `http://localhost:7887`. You should see the MoleculeMiner interface.

If you are running the server on a remote machine, you can access it by replacing `localhost` with the IP address of the machine, or by using an ssh tunnel/port forwarding. For example:

```shell
ssh -L 7887:localhost:7887 user@remote-machine
```

## Background & Usage

The server is designed to be a simple interface for running the MoleculeMiner pipeline on a PDF. The server will allow you to upload a PDF, run the pipeline on them, and download the results. The server will also store the results in a local database, so you can access them later. This is particularly useful for long-running jobs, as you can close the browser and come back later to download the results.

The app has two main pages: an upload page and a viewing/download page. The upload page allows you to upload a PDF, and the viewing/download page allows you to see the results of the pipeline and download them.

### Upload Page

On the upload page, there are three pieces of information you can provide:
1. A PDF file in the file upload box
2. An (optional) OpenAI API key in a text box
3. An (optional) checkbox to enable table extraction mode 

If #3 is checked, the pipeline will also extract tabular data from the PDF. Note that this takes significantly longer than the basic molecule extraction pipeline, as it requires communicating with the OpenAI API and then parsing the results. If a user only wants to extract molecule structures, they should leave this unchecked.
 
### Viewing/Download Page

On the viewing/download page, you will see a list of all the PDFs that have been uploaded to the server in a file selector. You can select a PDF from the list to view the results of the pipeline. Once a file is selected, a download link will appear. Additionally, the fill will be loaded into the viewer, which will show red bounding boxes overlaid on the PDF page to indicate where molecules were detected.
