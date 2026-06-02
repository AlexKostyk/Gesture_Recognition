const translateInput = document.getElementById("translateInput");
const translateButton = document.getElementById("translateButton");
const translateStatus = document.getElementById("translateStatus");
const translateStats = document.getElementById("translateStats");
const tokenList = document.getElementById("tokenList");
const variantNextButton = document.getElementById("variantNextButton");
const selectedTitle = document.getElementById("selectedTitle");
const selectedDescription = document.getElementById("selectedDescription");
const gestureVideo = document.getElementById("gestureVideo");
const sequenceInfo = document.getElementById("sequenceInfo");
const sequenceStatus = document.getElementById("sequenceStatus");
const sequenceList = document.getElementById("sequenceList");

let translatedTokens = [];
let selectedTokenIndex = -1;
let activeSequence = [];
let activeSequenceIndex = 0;
let autoSequenceEnabled = false;
let tokenVariantOffsets = [];

function setTranslateStatus(text, isError = false) {
  translateStatus.textContent = text;
  translateStatus.style.color = isError ? "#b3261e" : "#5c6773";
}

function clearPlayer() {
  gestureVideo.pause();
  gestureVideo.removeAttribute("src");
  gestureVideo.load();
  sequenceInfo.classList.add("hidden");
  sequenceStatus.textContent = "";
  sequenceList.innerHTML = "";
  activeSequence = [];
  activeSequenceIndex = 0;
  autoSequenceEnabled = false;
}

function renderSequenceBadges(sequence, currentIndex) {
  sequenceList.innerHTML = "";
  sequence.forEach((item, index) => {
    const badge = document.createElement("button");
    badge.type = "button";
    badge.className = `sequence-badge sequence-badge-button${
      index === currentIndex ? " sequence-badge-active" : ""
    }`;
    badge.textContent = item.label;
    badge.addEventListener("click", () => {
      autoSequenceEnabled = false;
      playSequenceAt(index);
    });
    sequenceList.appendChild(badge);
  });
}

function playSequenceAt(index) {
  if (!activeSequence.length || index < 0 || index >= activeSequence.length) {
    return;
  }

  activeSequenceIndex = index;
  const item = activeSequence[index];
  gestureVideo.src = item.video_url;
  gestureVideo.load();
  gestureVideo.play().catch(() => {});
  renderSequenceBadges(activeSequence, activeSequenceIndex);
  sequenceStatus.textContent = `Буква ${index + 1} из ${activeSequence.length}: ${item.label}`;
}

function renderTokens() {
  tokenList.innerHTML = "";

  translatedTokens.forEach((item, index) => {
    const button = document.createElement("button");
    const statusClass = item.mode === "dactyl" ? "token-chip-dactyl" : "token-chip-direct";
    const activeClass = index === selectedTokenIndex ? " token-chip-active" : "";

    button.type = "button";
    button.className = `token-chip ${statusClass}${activeClass}`;
    button.textContent = item.token;
    button.title = item.mode === "dactyl"
      ? "Готового видео слова нет, будет показано по буквам"
      : "Есть готовое видео жеста"

    button.addEventListener("click", () => {
      showToken(index);
    });

    tokenList.appendChild(button);
  });
}

function showToken(index) {
  const item = translatedTokens[index];
  if (!item) {
    return;
  }

  selectedTokenIndex = index;
  renderTokens();
  selectedTitle.textContent = item.token;

  if (item.found_direct) {
    selectedDescription.textContent = "Найдено готовое видео для слова.";
    clearPlayer();
    gestureVideo.src = item.sequence[0].video_url;
    gestureVideo.load();
    gestureVideo.play().catch(() => {});
    return;
  }

  if (!item.sequence.length) {
    clearPlayer();
    selectedDescription.textContent =
      item.missing_letters.length > 0
        ? `Для слова нет жеста и не найдены буквы: ${item.missing_letters.join(", ")}`
        : "Для слова нет готового жеста и не удалось собрать последовательность букв.";
    return;
  }

  selectedDescription.textContent = "Готового жеста нет, показываем слово по дактильным буквам.";
  sequenceInfo.classList.remove("hidden");
  activeSequence = item.sequence;
  autoSequenceEnabled = true;
  playSequenceAt(0);
}

async function translateText() {
  const text = translateInput.value.trim();
  if (!text) {
    setTranslateStatus("Сначала введите текст.", true);
    return;
  }

  translateButton.disabled = true;
  setTranslateStatus("Подбираем жесты и буквы...");
  translateStats.textContent = "";
  tokenList.innerHTML = "";
  translatedTokens = [];
  selectedTokenIndex = -1;
  tokenVariantOffsets = [];
  clearPlayer();
  selectedTitle.textContent = "Демонстрация";
  selectedDescription.textContent = "Нажмите на слово, чтобы запустить его жест.";

  try {
    const response = await fetch("/api/text-to-gesture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, variant_offset: 0 }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Ошибка сервера: ${response.status}`);
    }

    const data = await response.json();
    translatedTokens = Array.isArray(data.tokens) ? data.tokens : [];
    tokenVariantOffsets = translatedTokens.map((item) => Number(item.variant_offset ?? 0) || 0);
    renderTokens();

    const direct = data.stats?.direct_matches ?? 0;
    const dactyl = data.stats?.dactyl_fallbacks ?? 0;
    translateStats.textContent = `Слов: ${translatedTokens.length} | готовых жестов: ${direct} | по буквам: ${dactyl}`;

    if (!translatedTokens.length) {
      setTranslateStatus("Подходящие слова не найдены.", true);
      return;
    }

    setTranslateStatus("Нажмите на слово, чтобы запустить его демонстрацию.");
  } catch (error) {
    setTranslateStatus(`Не удалось подготовить жесты: ${error.message}`, true);
  } finally {
    translateButton.disabled = false;
  }
}

async function loadNextVariantForSelectedToken() {
  if (selectedTokenIndex < 0 || selectedTokenIndex >= translatedTokens.length) {
    return;
  }

  const currentItem = translatedTokens[selectedTokenIndex];
  const nextOffset = (tokenVariantOffsets[selectedTokenIndex] || 0) + 1;

  variantNextButton.disabled = true;
  setTranslateStatus(`Подбираем другой вариант для слова "${currentItem.token}"...`);

  try {
    const response = await fetch("/api/text-to-gesture/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        token: currentItem.token,
        variant_offset: nextOffset,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Ошибка сервера: ${response.status}`);
    }

    const data = await response.json();
    const updatedItem = data.item;
    translatedTokens[selectedTokenIndex] = updatedItem;
    tokenVariantOffsets[selectedTokenIndex] = Number(updatedItem.variant_offset ?? nextOffset) || nextOffset;

    renderTokens();
    showToken(selectedTokenIndex);
    setTranslateStatus(`Показан другой вариант для слова "${currentItem.token}".`);
  } catch (error) {
    setTranslateStatus(`Не удалось получить другой вариант: ${error.message}`, true);
  } finally {
    variantNextButton.disabled = false;
  }
}

gestureVideo.addEventListener("ended", () => {
  if (!activeSequence.length || !autoSequenceEnabled) {
    return;
  }

  const nextIndex = activeSequenceIndex + 1;
  if (nextIndex < activeSequence.length) {
    playSequenceAt(nextIndex);
  }
});

translateButton.addEventListener("click", () => {
  translateText();
});

translateInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    translateText();
  }
});

variantNextButton.addEventListener("click", () => {
  loadNextVariantForSelectedToken();
});
