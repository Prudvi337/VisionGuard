document.addEventListener('DOMContentLoaded', function() { 
    const imageUpload = document.getElementById('imageUpload');
    const preview = document.getElementById('preview');
    const testButton = document.getElementById('testButton');
    const modelSelect = document.getElementById('modelSelect');
    const resultsDisplay = document.getElementById('resultsDisplay');
    const downloadResults = document.getElementById('downloadResults');

    let uploadedImages = [];

    // Handle image uploads and preview
    imageUpload.addEventListener('change', function(event) {
        preview.innerHTML = '';
        uploadedImages = [];
        for (let file of event.target.files) {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'preview-image';
                    preview.appendChild(img);
                    uploadedImages.push({file: file, data: e.target.result});
                };
                reader.readAsDataURL(file);
            }
        }
    });

    // Handle testing the selected model
    testButton.addEventListener('click', function() {
        if (uploadedImages.length === 0) {
            alert('Please upload at least one image before testing.');
            return;
        }

        const selectedModel = modelSelect.value;
        resultsDisplay.innerHTML = ''; // Clear previous results

        uploadedImages.forEach((image, index) => {
            if (selectedModel === 'logoRecognition') {
                testNewLogoRecognitionModel(image, index);
            } else if (selectedModel === 'freshnessDetection') {
                testFreshnessDetection(image, index);
            }
        });
    });

    // Function to test the freshness detection model
    function testFreshnessDetection(image, index) {
        const formData = new FormData();
        formData.append('image', image.file);
        formData.append('produce_type', 'unknown'); // You can add user input for this

        fetch('/test_freshness', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const result = `
                <div class="result-item">
                    <img src="${image.data}" alt="Uploaded Image" class="result-image">
                    <div class="result-details">
                        <h3>Results for Image ${index + 1}</h3>
                        <p>Freshness: ${data.freshness}</p>
                        <p>Fresh Probability: ${(data.fresh_probability * 100).toFixed(2)}%</p>
                        <p>Rotten Probability: ${(data.rotten_probability * 100).toFixed(2)}%</p>
                        <p>Estimated Shelf Life: ${data.estimated_shelf_life} days</p>
                    </div>
                </div>
            `;
            resultsDisplay.innerHTML += result;
        })
        .catch(error => {
            console.error('Error:', error);
            resultsDisplay.innerHTML += `<p>Error processing image ${index + 1}: ${error.message}</p>`;
        });
    }

    // Function to test the new logo recognition model
    function testNewLogoRecognitionModel(image, index) {
        const formData = new FormData();
        formData.append('image', image.file);

        fetch('/test_logo', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const result = `
                <div class="result-item">
                    <img src="${image.data}" alt="Uploaded Image" class="result-image">
                    <div class="result-details">
                        <h3>Results for Image ${index + 1}</h3>
                        <p>Detected Logo: ${data.predicted_brand}</p>
                        <p>Predicted Count: ${data.predicted_count}</p>
                    </div>
                </div>
            `;
            resultsDisplay.innerHTML += result;
        })
        .catch(error => {
            console.error('Error:', error);
            resultsDisplay.innerHTML += `<p>Error processing image ${index + 1}: ${error.message}</p>`;
        });
    }

    // Function to download results as a text file
    downloadResults.addEventListener('click', function() {
        const results = resultsDisplay.innerText;
        const blob = new Blob([results], {type: 'text/plain'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'visionguard_results.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
});
