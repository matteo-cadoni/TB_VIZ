import debounce from "just-debounce-it";
import * as vizarr from "./src/index";

async function main() {
  console.log(`vizarr v${vizarr.version}: https://github.com/hms-dbmi/vizarr`);

  const viewer = await vizarr.createViewer(document.querySelector("#root")!);
  const url = new URL(window.location.href);
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;


  let sourceUrls = url.searchParams.getAll("source");

  if (sourceUrls.length === 0) {
    const inputDiv = document.createElement('div');
    inputDiv.style.position = 'fixed';
    inputDiv.style.top = '50%';
    inputDiv.style.left = '55%';
    inputDiv.style.transform = 'translate(-50%, -50%)';  // Centers the div
    inputDiv.style.zIndex = '1000';
    inputDiv.style.backgroundColor = 'white';
    inputDiv.style.padding = '20px';
    inputDiv.style.borderRadius = '5px';
    inputDiv.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.1)';  // Optional shadow for better visibility

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.style.padding = '10px';
    fileInput.style.fontSize = '16px';

    const uploadButton = document.createElement('button');
    uploadButton.textContent = 'Upload CZI File';
    uploadButton.style.padding = '10px';
    uploadButton.style.fontSize = '16px';
    uploadButton.style.marginLeft = '10px';

    inputDiv.appendChild(fileInput);
    inputDiv.appendChild(uploadButton);
    document.body.appendChild(inputDiv);

    // Create loading spinner element



    uploadButton.addEventListener("click", async () => {
      const file = fileInput.files?.[0];
      if (!file) {
        alert("Please select a file first.");
        return;
      }

      const loadingDiv = document.createElement('div');
      loadingDiv.style.position = 'fixed';
      loadingDiv.style.top = '50%';
      loadingDiv.style.left = '50%';
      loadingDiv.style.transform = 'translate(-50%, -50%)';
      loadingDiv.style.zIndex = '1001'; // Behind the inputDiv
      loadingDiv.style.display = 'none'; // Initially hidden
      loadingDiv.style.textAlign = 'center';
      loadingDiv.style.display = 'flex';
      loadingDiv.style.flexDirection = 'column';  // Stack spinner and text vertically
      loadingDiv.style.alignItems = 'center';  // Horizontally center the items
      loadingDiv.style.justifyContent = 'center';
      loadingDiv.style.backgroundColor = 'rgba(0, 0, 0, 0.6)'; // Add semi-transparent background for better contrast
      loadingDiv.style.padding = '20px';
      loadingDiv.style.borderRadius = '10px';

      const spinner = document.createElement('div');
      spinner.style.border = '4px solid #f3f3f3';
      spinner.style.borderTop = '4px solid #3498db';
      spinner.style.borderRadius = '50%';
      spinner.style.width = '40px';
      spinner.style.height = '40px';
      spinner.style.animation = 'spin 2s linear infinite';

      const loadingText = document.createElement('p');
      loadingText.textContent = "Processing image, this could take some minutes...";
      loadingText.style.marginTop = '10px';
      loadingText.style.fontSize = '16px';
      loadingText.style.color = '#fff';

      loadingDiv.appendChild(spinner);
      loadingDiv.appendChild(loadingText);
      document.body.appendChild(loadingDiv);

      // Show the loading spinner and text
      loadingDiv.style.display = 'flex';

      const formData = new FormData();
      formData.append("file", file);

      try {
        const uploadResponse = await fetch(`${BACKEND_URL}/upload_czi`, {
          method: "POST",
          body: formData,
        });
        if (!uploadResponse.ok) throw new Error("Upload failed");

        const uploadData = await uploadResponse.json();
        const filePath = uploadData.file_path;

        const processResponse = await fetch(
          `${BACKEND_URL}/get_bacilli_count/?file_path=${encodeURIComponent(filePath)}`
        );
        if (!processResponse.ok) throw new Error("Processing failed");

        const processData = await processResponse.json();
        sourceUrls = [processData.og_path, processData.map_path].filter(Boolean);

        if (sourceUrls.length === 0) throw new Error("No valid source URLs received");

        console.log("File processed, moving image directories...");
        console.log(sourceUrls);

        for (const sourceUrl of sourceUrls) {
          const moveResponse = await fetch(`${BACKEND_URL}/move-image`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              filename: sourceUrl,
              target_dir: "../TB_Frontend/dist",
            }),
          });
          if (!moveResponse.ok) throw new Error(`Moving image failed for ${sourceUrl}`);
          console.log(`File moved, loading Zarr image from: ${sourceUrl}`);
        }

        url.searchParams.delete("source");
        sourceUrls.forEach(src => url.searchParams.append("source", new URL(src, window.location.origin).href));
        window.history.pushState({}, "", decodeURIComponent(url.href));

        console.log("URL updated:", decodeURIComponent(url.href));
        loadImages(sourceUrls);
        inputDiv.remove();

        // Hide the loading spinner and text after processing is done
        loadingDiv.style.display = 'none';
      } catch (error) {
        console.error("Error processing file:", error);
        alert("File processing failed. Please try again.");

        // Hide the loading spinner and text in case of error
        loadingDiv.style.display = 'none';
      }
    });
  } else {
    loadImages(sourceUrls);
  }

  function loadImages(sourceUrls: string[]) {
  const viewStateString = url.searchParams.get("viewState");
  if (viewStateString) {
    const viewState = JSON.parse(viewStateString);
    viewer.setViewState(viewState);
  }

  viewer.on(
    "viewStateChange",
    debounce((update: any) => {  // Change 'any' to a more specific type if possible
      const url = new URL(window.location.href);
      url.searchParams.set("viewState", JSON.stringify(update));
      window.history.pushState({}, "", decodeURIComponent(url.href));
    }, 200)
  );

  const sources: string[] = url.searchParams.getAll("source");

  sources.forEach((source: string) => {
    const config :vizarr.ImageLayerConfig = { source: decodeURIComponent(source),
                                            name: source.split("_")[0].split("/").at(-1)};
    viewer.addImage(config);
  });
}
}



// Add CSS for the spinning animation (This can be placed in a separate CSS file too)
const style = document.createElement('style');
style.innerHTML = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);

main();
