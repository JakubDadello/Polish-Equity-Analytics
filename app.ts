function updateProgressBars (results: any) {
    const high = results.high;
    const middle = results.middle;
    const low = results.low;

    (document.getElementById("fill-high") as HTMLElement)!.style.width = high + "%";
    (document.getElementById("fill-high") as HTMLElement)!.textContent = high + "%";

    (document.getElementById("fill-middle") as HTMLElement)!.style.width = middle + "%";
    (document.getElementById("fill-middle") as HTMLElement)!.textContent = middle + "%";

    (document.getElementById("fill-low") as HTMLElement)!.style.width = low + "%";
    (document.getElementById("fill-low") as HTMLElement)!.textContent = low + "%";
}


async function predict () {
    const values = [
        Number((document.getElementById("net_income") as HTMLInputElement).value),
        Number((document.getElementById("net_cash_flow") as HTMLInputElement).value),
        Number((document.getElementById("roe") as HTMLInputElement).value),
        Number((document.getElementById("roa") as HTMLInputElement).value),
        Number((document.getElementById("ebitda") as HTMLInputElement).value),
        String((document.getElementById("sector") as HTMLInputElement).value),
        Number((document.getElementById("cumulation") as HTMLInputElement).value)
    ]

    try {
        const res = await fetch ("https://<api-id>.execute-api.eu-central-1.amazonaws.com/development/prediction", 
        {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({data: [values]})
        })

        if(!res.ok) {
            throw new Error("Failed to load model")
        }

        const results = await res.json();

        updateProgressBars(results); 

        } catch (error) {
           throw new Error ("Failed to load model")
        }

    
}