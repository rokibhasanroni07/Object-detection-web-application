// Load the COCO-SSD model
cocoSsd.load().then(model => {
    console.log('Model loaded');

    document.getElementById('imageUpload').addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (file) {
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = () => {
                detectObjects(model, img);
            };
        }
    });

    document.getElementById('webcamButton').addEventListener('click', () => {
        startWebcam(model);
    });
});

// Function to detect objects in an image
function detectObjects(model, img) {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    const canvasWidth = 800; // Example width
    const canvasHeight = 600; // Example height
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    const scale = Math.min(canvasWidth / img.width, canvasHeight / img.height);
    const scaledWidth = img.width * scale;
    const scaledHeight = img.height * scale;
    const offsetX = (canvasWidth - scaledWidth) / 2;
    const offsetY = (canvasHeight - scaledHeight) / 2;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight); // Clear the canvas
    ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

    model.detect(img).then(predictions => {
        console.log('Predictions:', predictions);

        predictions.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox.map((value, index) => {
                return index % 2 === 0 // For x and width
                    ? value * scale + offsetX
                    : value * scale + offsetY; // For y and height
            });

            ctx.strokeStyle = 'black';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);

            // Set font size and style
            ctx.font = '20px Arial'; // Adjust font size (e.g., 16px)
            ctx.fillStyle = 'black';
            ctx.fillText(
                `${prediction.class} (${Math.round(prediction.score * 100)}%)`,
                x,
                y > 10 ? y - 5 : 10
            );
        });
    });
}



// Function to start webcam and detect objects in real-time
function startWebcam(model) {
    const video = document.createElement('video'); // Create a hidden video element

    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream; // Assign webcam stream to video
        video.play(); // Start the video feed

        video.addEventListener('loadeddata', () => {
            console.log('Webcam loaded');
            detectFrame(model, video); // Start object detection
        });
    });
}

function detectFrame(model, video) {
    const canvas = document.getElementById('canvas'); // Use only the canvas for output
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    async function detect() {
        // Draw the video frame onto the canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Perform object detection
        const predictions = await model.detect(video);

        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw the video frame again after clearing
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Draw bounding boxes and labels
        predictions.forEach(prediction => {
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 3;
            ctx.strokeRect(...prediction.bbox); // Draw bounding box
            ctx.fillStyle = 'black';
            ctx.fillText(
                `${prediction.class} (${Math.round(prediction.score * 100)}%)`,
                prediction.bbox[0],
                prediction.bbox[1] > 10 ? prediction.bbox[1] - 5 : 10
            );
        });

        // Request the next frame for detection
        requestAnimationFrame(detect);
    }

    detect();
}

