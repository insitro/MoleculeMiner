document.getElementById('file-select').addEventListener('change', async (e) => {
    const selectedFile = encodeURIComponent(e.target.value);

    if (selectedFile) {
        // Clean up previous blob URL, if one exists
        const downloadButton = document.getElementById('download-button');
        if (downloadButton.href) {
            window.URL.revokeObjectURL(downloadButton.href);
        }

        try {
            const response = await fetch('/view_file?filename=' + selectedFile);

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();

            // Process and display the file
            process_file(result.pdf_file, result.csv_file);

        } catch (error) {
            console.error('Error:', error);
        }
    }
});

const pdfRenderer = document.getElementById('pdf-page');
const modal = document.getElementById('info-modal');

function process_file(pdf, csv) {
    create_download_button(csv);
    const result = process_csv(csv);
    const table_data = result.table_data;
    const refs = result.refs;
    loadPDF(pdf, table_data, refs);
}

function create_download_button(csv) {
    const downloadButton = document.getElementById('download-button');
    if (downloadButton) {
        // Create blob from CSV data
        const blob = new Blob([csv], {type: 'text/csv'});
        downloadButton.href = window.URL.createObjectURL(blob);
        downloadButton.style.display = 'block';

        const selectedFile = document.getElementById('file-select').value;
        const baseName = selectedFile.split('.')[0];
        downloadButton.download = `${baseName}_results.tsv`;
    }
}

function process_csv(csv) {
    const table_data = {};
    let refs = null;

    Papa.parse(csv, {
        header: false,           // Use first row as header
        dynamicTyping: false,    // Automatically convert types
        delimiter: "\t",        // Tab character for TSV
        skipEmptyLines: true,   // Skip any empty Lines
        complete: function (results) {
            const data = results.data;
            results_final = data;
            const rows = results_final.length;
            let full_id = null;
            // Parse each row to get the SMILES info and the table data if any
            // Skip the header row but record the headers
            refs = results_final[0];
            for (let i = 1; i < rows; i++) {
                // These indexes correspond to respective columns in the TSV file
                let rowPage = parseInt(results_final[i][2], 10);
                let mol_id = results_final[i][3];
                full_id = rowPage.toString() + '_' + mol_id;
                if (!(rowPage in table_data)) {
                    table_data[rowPage] = [results_final[i].slice(1)];
                } else {
                    // Page already exists in dict; push to page list
                    table_data[rowPage].push(results_final[i].slice(1));
                }
            }
            

        },
        error: function (error) {
            console.error('Error parsing TSV file:', error);
        }
    });
    return {table_data:table_data, refs:refs};
}

async function loadPDF(pdfData, table_data, refs) {
    const pdfBytes = atob(pdfData);
    // Get all the page keys to render from table_data
    const pages = Object.keys(table_data).map(key => parseInt(key, 10));

    const desiredScale = 100 / 96;
    const pdfDoc = await pdfjsLib.getDocument({data: pdfBytes}).promise;
    pdfRenderer.innerHTML = '';

    for (let i = 0; i < pages.length; i++) {
        const page = await pdfDoc.getPage(pages[i]);
        const viewport = page.getViewport({scale: desiredScale});

        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const context = canvas.getContext('2d');

        const pageContainer = document.createElement('div');
        pageContainer.className = 'pdf-page';
        pageContainer.style.width = `${viewport.width}px`;
        pageContainer.style.height = `${viewport.height}px`;
        pageContainer.appendChild(canvas);
        pdfRenderer.appendChild(pageContainer);

        await page.render({canvasContext: context, viewport: viewport}).promise;

        // Grab the (1) box rows & (2) Metadata if any; for the page
        const pageBoxRows = table_data[pages[i]];
        const pageBoxMeta = refs;
        

        createBoundingBoxes(pageContainer, pageBoxRows, pageBoxMeta, viewport);
    }

}

function createBoundingBoxes(pageContainer, pageBoxRows, pageBoxMeta, viewport) {

    for (let i = 0; i < pageBoxRows.length; i++) {
        const div = document.createElement('div');
        div.className = 'bounding-box';
        // These indexes correspond to respective columns in the TSV file,
        // minus the first "PDF" column
        const x1 = Math.floor(parseInt(pageBoxRows[i][3]) * 0.25);
        const y1 = Math.floor(parseInt(pageBoxRows[i][4]) * 0.25);
        const x2 = Math.floor(parseInt(pageBoxRows[i][5]) * 0.25);
        const y2 = Math.floor(parseInt(pageBoxRows[i][6]) * 0.25);
        const w = x2 - x1;
        const h = y2 - y1;
        div.style.left = `${x1}px`;
        div.style.top = `${y1}px`;
        div.style.width = `${w}px`;
        div.style.height = `${h}px`;

        const molID = pageBoxRows[i][1] + '_' + pageBoxRows[i][2];

        const metaInfo = pageBoxMeta;
        
        div.onclick = (e) => {
            e.stopPropagation();
            showModal(div, pageBoxRows[i], metaInfo);
        }

        pageContainer.appendChild(div);
    }
}

function showModal(div, boxinfo, metainfo) {
    // Remove existing modal if any
    const existingModal = document.getElementById(boxinfo[1] + '-' + boxinfo[2]);
    if (existingModal) {
        existingModal.remove();
    }

    // Create the first div by paragraphs
    const labels = ['Page No.: ', 'SMILES: ', 'Model Confidence: '];
    const posids = [1, 7, 8];
    // Modal Construction
    const mainModal = document.createElement('div')
    mainModal.className = 'modal-wrapper';
    mainModal.id = boxinfo[1] + '-' + boxinfo[2];
    mainModal.style.display = 'flex';
    const contentModal = document.createElement('div');
    contentModal.className = 'modal-content';
    // contentModal.id = boxinfo[1]+'-'+boxinfo[2];
    const btnModal = document.createElement('button');
    btnModal.className = 'close-btn';
    btnModal.textContent = 'Close';
    for (let i = 0; i < posids.length; i++) {
        // Adding Paragraphs
        const pageinfo = document.createElement('p');
        pageinfo.className = 'box-metadata';
        pageinfo.textContent = labels[i] + boxinfo[posids[i]];
        contentModal.appendChild(pageinfo);
        
    }

    // Add Metadata if any
    if (boxinfo.length > 9) {
        for (let j=9; j<boxinfo.length; j++) {
        const metaDiv =  document.createElement('p');
        metaDiv.className = 'box-metadata';
        metaDiv.textContent = metainfo[j+1] + ': ' + boxinfo[j];
        contentModal.appendChild(metaDiv);
        }
    }
    

    // Add the close button
    contentModal.appendChild(btnModal);

    // contentModal.onclick = () => closeModal(boxinfo[1]+'-'+boxinfo[2]);
    btnModal.onclick = (e) => {
        e.stopPropagation(); // Stop the click from reaching the div underneath
        closeModal(boxinfo[1] + '-' + boxinfo[2]);
    }
    mainModal.appendChild(contentModal);

    // div.appendChild(mainModal);
    document.body.appendChild(mainModal);

    // Position the modal relative to the bounding box
    const divRect = div.getBoundingClientRect();
    mainModal.style.left = `${divRect.left + window.scrollX}px`;
    mainModal.style.top = `${divRect.top + window.scrollY}px`;
}

function closeModal(box_id) {
    const boxDiv = document.getElementById(box_id);
    boxDiv.remove();
}
