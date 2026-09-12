// Chữ Nôm Camera OCR - capture + review + translate flow.
// Matches api/routers/ocr.py's request/response shape exactly:
//   POST /v1/ocr (multipart/form-data): file, rescore, translate, translate_line_ids
//   (comma-separated ints), max_lines_to_translate
//   -> OcrResponse { request_id, detection_status, lines: LineResult[], line_count,
//                    translated_count, translation_truncated, warnings }

const photoInput = document.getElementById("photo-input");
const preview = document.getElementById("preview");
const statusSection = document.getElementById("status-section");
const statusText = document.getElementById("status-text");
const resultsSection = document.getElementById("results-section");
const resultsSummary = document.getElementById("results-summary");
const lineList = document.getElementById("line-list");
const translateControls = document.getElementById("translate-controls");
const selectAllBtn = document.getElementById("select-all-btn");
const translateBtn = document.getElementById("translate-btn");

// Kept in JS state so it can be POSTed a second time (translate) without recapturing.
let capturedFile = null;
let lastLines = [];

function setStatus(text) {
  statusSection.hidden = !text;
  statusText.textContent = text || "";
}

function resetResults() {
  resultsSection.hidden = true;
  lineList.innerHTML = "";
  translateControls.hidden = true;
  lastLines = [];
}

photoInput.addEventListener("change", async () => {
  const file = photoInput.files[0];
  if (!file) return;
  capturedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  resetResults();

  setStatus("Đang nhận dạng... / Recognizing...");
  try {
    const formData = new FormData();
    formData.append("file", capturedFile);
    formData.append("translate", "false");
    const resp = await fetch(`${API_BASE_URL}/v1/ocr`, { method: "POST", body: formData });
    if (!resp.ok) {
      setStatus(`Lỗi (${resp.status}): ${await resp.text()}`);
      return;
    }
    const data = await resp.json();
    setStatus("");
    renderResults(data);
  } catch (err) {
    setStatus(`Kết nối thất bại: ${err}`);
  }
});

function renderResults(data) {
  resultsSection.hidden = false;
  lastLines = data.lines;

  if (data.detection_status === "no_lines_found") {
    resultsSummary.textContent = "Không tìm thấy chữ nào trong ảnh - hãy thử chụp lại.";
    return;
  }

  resultsSummary.textContent = `Tìm thấy ${data.line_count} dòng chữ.`;
  if (data.warnings && data.warnings.length) {
    resultsSummary.textContent += ` (${data.warnings.length} cảnh báo)`;
  }

  lineList.innerHTML = "";
  for (const line of data.lines) {
    lineList.appendChild(renderLineItem(line));
  }
  translateControls.hidden = !data.lines.some((l) => l.text != null);
}

function renderLineItem(line) {
  const li = document.createElement("li");
  li.className = "line-item" + (line.recognition_error ? " has-error" : "");
  li.dataset.lineId = line.line_id;

  const row = document.createElement("div");
  row.className = "line-row";

  if (line.text != null) {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "line-checkbox";
    row.appendChild(checkbox);
  }

  const textSpan = document.createElement("span");
  textSpan.className = "line-text";
  textSpan.textContent = line.text ?? "(nhận dạng thất bại)";
  row.appendChild(textSpan);
  li.appendChild(row);

  if (line.recognition_error) {
    const err = document.createElement("p");
    err.className = "line-error";
    err.textContent = line.recognition_error;
    li.appendChild(err);
  }

  if (line.translation_status && line.translation_status !== "skipped") {
    li.appendChild(renderTranslation(line));
  }

  return li;
}

function renderTranslation(line) {
  const box = document.createElement("div");
  box.className = "line-translation";

  if (line.reading) {
    const reading = document.createElement("p");
    reading.className = "line-reading";
    reading.textContent = line.reading;
    box.appendChild(reading);
  }

  const statusLabel = {
    ok: null,
    empty: "Không có bản dịch (thử lại không thành công) / No translation (retry still empty)",
    error: "Dịch thất bại / Translation failed",
  }[line.translation_status];

  if (line.translation) {
    const translation = document.createElement("p");
    translation.className = "line-translation-text";
    translation.textContent = line.translation;
    box.appendChild(translation);
  }
  if (statusLabel) {
    const status = document.createElement("p");
    status.className = "line-translation-status" + (line.translation_status === "error" ? " status-error" : "");
    status.textContent = statusLabel;
    box.appendChild(status);
  }
  return box;
}

selectAllBtn.addEventListener("click", () => {
  for (const checkbox of lineList.querySelectorAll(".line-checkbox")) {
    checkbox.checked = true;
  }
});

translateBtn.addEventListener("click", async () => {
  const checkedIds = [...lineList.querySelectorAll(".line-item")]
    .filter((li) => li.querySelector(".line-checkbox")?.checked)
    .map((li) => li.dataset.lineId);

  if (checkedIds.length === 0) {
    setStatus("Chọn ít nhất một dòng để dịch. / Select at least one line to translate.");
    return;
  }
  if (!capturedFile) return;

  setStatus("Đang dịch... / Translating...");
  try {
    const formData = new FormData();
    formData.append("file", capturedFile);
    formData.append("translate", "true");
    formData.append("translate_line_ids", checkedIds.join(","));
    const resp = await fetch(`${API_BASE_URL}/v1/ocr`, { method: "POST", body: formData });
    if (!resp.ok) {
      setStatus(`Lỗi (${resp.status}): ${await resp.text()}`);
      return;
    }
    const data = await resp.json();
    setStatus(data.translation_truncated ? "Đã dịch (một số dòng bị giới hạn) / Translated (some lines were capped)" : "");
    renderResults(data);
  } catch (err) {
    setStatus(`Kết nối thất bại: ${err}`);
  }
});
