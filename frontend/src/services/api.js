export const analyzeLake = async (satelliteFile, demFile) => {
    const form = new FormData();
    form.append("satellite", satelliteFile);
    if (demFile) {
        form.append("dem", demFile);
    }

    console.log("[DEBUG] Sending request to backend...");
    try {
        const res = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            body: form,
        });

        console.log("[DEBUG] Response status:", res.status);

        if (!res.ok) {
            const errText = await res.text();
            console.error("[DEBUG] Error response:", errText);
            throw new Error(`Error: ${res.status} ${res.statusText} - ${errText}`);
        }

        const data = await res.json();
        console.log("[DEBUG] Response data:", data);
        return data;
    } catch (error) {
        console.error("Analysis failed:", error);
        throw error;
    }
};
