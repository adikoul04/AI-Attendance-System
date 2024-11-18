document.getElementById("start-dataset").addEventListener("click", () => {
    const name = document.getElementById("name").value;

    if (!name) {
        showResponse("Please enter a name.", "error");
        return;
    }

    fetch("/start_dataset", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: name }),
    })
        .then((response) => response.json())
        .then((data) => {
            showResponse(`Dataset recording started for: ${data.name}`, "success");
        })
        .catch((error) => {
            showResponse(`Error: ${error.message}`, "error");
        });
});

document.getElementById("start-attendance").addEventListener("click", () => {
    fetch("/start_attendance", {
        method: "POST",
    })
        .then((response) => response.json())
        .then((data) => {
            showResponse("Attendance tracking started!", "success");
        })
        .catch((error) => {
            showResponse(`Error: ${error.message}`, "error");
        });
});

function showResponse(message, type) {
    const responseDiv = document.getElementById("response");
    responseDiv.innerText = message;
    responseDiv.style.color = type === "success" ? "#ffffff" : "#ff4d4f";
    responseDiv.style.opacity = 0;
    setTimeout(() => {
        responseDiv.style.opacity = 1;
        responseDiv.style.transition = "opacity 0.5s ease";
    }, 100);
}
