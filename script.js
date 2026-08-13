document.getElementById("newsImage").addEventListener("change", function(e) {
    const fileName = e.target.files[0] ? e.target.files[0].name : "Upload Image for Tampering Check";
    document.querySelector(".file-upload-text").textContent = fileName;
});

document.getElementById("verifyForm").addEventListener("submit", async function(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    
    const btn = document.getElementById("submitBtn");
    const btnText = document.querySelector(".btn-text");
    const loader = document.querySelector(".loader");
    const resultsContainer = document.getElementById("resultsContainer");
    
    btn.disabled = true;
    btnText.style.display = "none";
    loader.style.display = "block";
    resultsContainer.classList.add("hidden");
    
    try {
        const response = await fetch("/api/verify", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) throw new Error("Server error");
        
        const data = await response.json();
        
        const verdictTag = document.getElementById("finalVerdictTag");
        verdictTag.textContent = data.final_verdict;
        verdictTag.style.backgroundColor = data.verdict_color;
        verdictTag.style.color = "#fff";
        verdictTag.style.boxShadow = `0 0 15px ${data.verdict_color}80`;
        
        const isLinguisticallyFake = data.linguistic_score.is_fake;
        const isEvidencePro = data.evidence_is_pro;
        const hasEvidence = data.evidence.length > 0;
        
        const logicFake = document.getElementById("logicFake");
        const logicStance = document.getElementById("logicStance");
        const logicSummary = document.getElementById("logicSummary");
        
        logicFake.textContent = isLinguisticallyFake ? "Failed (Looks Deceptive)" : "Passed (Looks Professional)";
        logicFake.style.color = isLinguisticallyFake ? "var(--danger)" : "var(--success)";
        
        if (!hasEvidence) {
            logicStance.textContent = "Unknown (No News Found)";
            logicStance.style.color = "var(--text-muted)";
            logicSummary.textContent = "Meaning: We couldn't find live news to fact-check this, so we are relying purely on linguistic analysis.";
        } else {
            logicStance.textContent = isEvidencePro ? "Passed (News Agrees)" : "Failed (News Disagrees)";
            logicStance.style.color = isEvidencePro ? "var(--success)" : "var(--danger)";
            
            if (!isLinguisticallyFake && isEvidencePro) {
                logicSummary.textContent = "Meaning: The text is professionally written and is backed up by live news. This is verified true.";
            } else if (isLinguisticallyFake && !isEvidencePro) {
                logicSummary.textContent = "Meaning: The text is highly deceptive and live news completely contradicts it. This is verified fake.";
            } else if (isLinguisticallyFake && isEvidencePro) {
                logicSummary.textContent = "Meaning: Live news confirms this event happened, but the text you provided is written in a highly deceptive, sensationalist, or click-bait manner.";
            } else if (!isLinguisticallyFake && !isEvidencePro) {
                logicSummary.textContent = "Meaning: The text is written very professionally (like a real news article), but live news proves that it is factually incorrect.";
            }
        }
        
        const fProb = (data.linguistic_score.prob_fake * 100).toFixed(1);
        const rProb = (data.linguistic_score.prob_real * 100).toFixed(1);
        
        document.getElementById("fakeProbVal").textContent = `${fProb}%`;
        document.getElementById("fakeProbFill").style.width = `${fProb}%`;
        document.getElementById("realProbVal").textContent = `${rProb}%`;
        document.getElementById("realProbFill").style.width = `${rProb}%`;
        
        const imgInd = document.getElementById("imageStatusIndicator");
        const imgMsg = document.getElementById("imageStatusMessage");
        imgMsg.textContent = data.image_analysis.message;
        
        if (data.image_analysis.status === "none") {
            imgInd.style.backgroundColor = "gray";
        } else if (data.image_analysis.tampered_prob < 0.5) {
            imgInd.style.backgroundColor = "var(--success)";
        } else {
            imgInd.style.backgroundColor = "var(--danger)";
        }
        
        const evidenceList = document.getElementById("evidenceList");
        evidenceList.innerHTML = "";
        
        if (data.evidence.length === 0) {
            evidenceList.innerHTML = `<p style="color: var(--text-muted); font-style: italic;">No related live news articles found to cross-reference.</p>`;
        } else {
            data.evidence.forEach(item => {
                const isPro = item.stance === "PRO";
                const div = document.createElement("div");
                div.className = `evidence-item ${isPro ? "pro" : "con"}`;
                
                div.innerHTML = `
                    <div class="evidence-source">${item.source}</div>
                    <div class="evidence-snippet">"${item.snippet}"</div>
                    <div class="evidence-stance-badge ${isPro ? "badge-pro" : "badge-con"}">
                        Stance: ${item.stance} (${(item.confidence * 100).toFixed(1)}% Confidence)
                    </div>
                `;
                evidenceList.appendChild(div);
            });
        }
        
        resultsContainer.classList.remove("hidden");
        
    } catch (error) {
        alert("An error occurred during verification. Make sure the backend is running.");
        console.error(error);
    } finally {
        btn.disabled = false;
        btnText.style.display = "block";
        loader.style.display = "none";
    }
});
