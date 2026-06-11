document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("uploadForm");
    const videoInput = document.getElementById("videoInput");
    const fileLabel = document.getElementById("fileLabel");
    const loadingBox = document.getElementById("loadingBox");
    const uploadZone = document.querySelector(".upload-zone");

    if (videoInput && fileLabel) {
        videoInput.addEventListener("change", function () {
            if (videoInput.files.length > 0) {
                fileLabel.textContent = videoInput.files[0].name;
            }
        });
    }

    if (uploadForm && loadingBox) {
        uploadForm.addEventListener("submit", function () {
            loadingBox.classList.remove("d-none");
        });
    }

    if (uploadZone && videoInput) {
        uploadZone.addEventListener("dragover", function (event) {
            event.preventDefault();
            uploadZone.classList.add("dragover");
        });

        uploadZone.addEventListener("dragleave", function () {
            uploadZone.classList.remove("dragover");
        });

        uploadZone.addEventListener("drop", function (event) {
            event.preventDefault();
            uploadZone.classList.remove("dragover");

            if (event.dataTransfer.files.length > 0) {
                videoInput.files = event.dataTransfer.files;

                if (fileLabel) {
                    fileLabel.textContent = event.dataTransfer.files[0].name;
                }
            }
        });
    }
});