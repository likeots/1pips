const formatPct = (value) =>
  typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(2)}%`
    : "—";

const setText = (id, text) => {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = text;
  }
};

const updateDirectionTable = (breakdown = {}) => {
  const tbody = document.getElementById("direction-body");
  if (!tbody) return;

  const entries = [
    ["Лонг", breakdown.long],
    ["Шорт", breakdown.short],
  ];

  const hasData = entries.some(([, data]) => data && data.trades);
  if (!hasData) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty">Нет данных</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  for (const [label, data] of entries) {
    const row = document.createElement("tr");
    if (!data || !data.trades) {
      row.innerHTML = `<td>${label}</td><td>0</td><td>—</td><td>—</td>`;
    } else {
      row.innerHTML = `
        <td>${label}</td>
        <td>${data.trades}</td>
        <td>${formatPct(data.base_win_rate)}</td>
        <td>${formatPct(data.plus1_win_rate)}</td>
      `;
    }
    tbody.appendChild(row);
  }
};

const showStatus = (message, isError = false) => {
  const statusCard = document.getElementById("data-status");
  const statusMessage = document.getElementById("status-message");
  if (!statusCard || !statusMessage) return;
  statusCard.style.display = "block";
  statusMessage.textContent = message;
  statusMessage.className = isError ? "error" : "";
};

fetch("data/one_pip_summary.json", { cache: "no-cache" })
  .then((response) => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  })
  .then((summary) => {
    setText("trades-kept", summary.trades_kept ?? "—");
    setText("base-wr", formatPct(summary.base?.win_rate));
    setText("plus-wr", formatPct(summary.plus1?.win_rate));
    setText("delta-wr", formatPct(summary.delta_win_rate));
    setText("p-value", summary.mcnemar?.p_value?.toFixed(6) ?? "—");

    const filters = summary.filters ?? {};
    setText("min-rr", filters.min_rr ?? "—");
    setText("min-sl", filters.min_sl_pips ?? "—");
    setText("min-tp", filters.min_tp_pips ?? "—");
    setText("tp-mode", filters.tp_mode ?? "—");

    updateDirectionTable(summary.direction_breakdown);

    showStatus("Данные успешно загружены.");
  })
  .catch((error) => {
    console.error(error);
    showStatus("Не удалось загрузить данные. Запустите симулятор, чтобы обновить файл.", true);
  });
