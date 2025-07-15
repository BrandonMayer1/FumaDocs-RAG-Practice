import { ChromaClient } from 'chromadb';
import { OllamaEmbeddings } from '@langchain/ollama';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { readFile } from 'fs/promises'; 
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';

//Create Ollama and ChromaClient collections
const client = new ChromaClient();
const embeddings = new OllamaEmbeddings({
    model: "mxbai-embed-large", 
    baseUrl: "http://localhost:11434", 
    });



function getFilePaths(dirPath, files = []) {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
      
    for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);
        if (entry.isDirectory()) {
            getFilePaths(fullPath, files); 
        } 
        else if (entry.isFile()) {
            files.push(fullPath);
        }
    }
    return files;
}


async function sumarizeOpenAI(yamlContent){
    const message = [
        {
            role: 'system',
            content: 
                ` You are tasked with generating a summary of an OpenAPI YAML file in markdown format. The YAML content will be provided to you, and you should create a concise yet informative summary that captures the essential elements of the API specification.
                Please follow these instructions to create the summary:

                1. Analyze the provided YAML content carefully.
                2. Create a markdown summary that includes the following sections:
                a. API Overview
                b. Server Information
                c. Authentication Methods
                d. Available Endpoints (grouped by tags)
                e. Schema Definitions (if present)

                3. For the API Overview, include:
                - API title
                - Version
                - Description (if provided)
                - Terms of service (if available)
                - Contact information (if available)
                - License information (if available)

                4. For Server Information, list the available servers and their URLs.

                5. For Authentication Methods, describe the security schemes used (e.g., API key, OAuth2).

                6. For Available Endpoints:
                - Group endpoints by their tags
                - For each endpoint, provide:
                    * HTTP method
                    * Path
                    * Summary or description
                    * Parameters (if any)
                    * Request body (if applicable)
                    * Responses (focus on successful responses, mention error responses briefly)

                7. For Schema Definitions, list the main data models used in the API, with a brief description of each.

                8. Use appropriate markdown formatting:
                - Use headers (##, ###, ####) to structure the document
                - Use bullet points or numbered lists for clarity
                - Use code blocks for examples of endpoints or schema snippets
                - Use tables where appropriate (e.g., for parameters or response codes)

                9. Keep the summary concise but informative. Focus on the most important aspects of the API.

                10. Ensure the markdown is well-formatted and easy to read.

                Please provide your markdown summary within <summary> tags. Begin your summary now.`
        },
        {
            role: 'user',
            content: 'Here is the Yaml:' + yamlContent
        }
    ];
      
    const topicPayload = {
        model: "deepseek-v2:latest",
        messages: message,
        stream: false,
    }

    try {
        const response = await fetch('http://localhost:11434/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(topicPayload)
        });
        const data = await response.json();
        return data.message.content;
    }
    catch (error){
        console.log("ERROR: " + error.message);
        throw error;
    }
}


async function storeInChroma(embedding, text, documentName) {
    try{
        const collection = await client.getOrCreateCollection({name: 'markdown-store'});
        await collection.upsert({
            ids: [Date.now().toString()],
            embeddings: [embedding],
            metadatas: [{name: documentName }],
            documents: [text]
        });
        console.log("|--STORED IN CHROMADB--|");
    }
    catch (error){
        console.log(error);
    }
}


async function headerChunking(text){
    const Seperators = [
        "\n# ", "\n## ", //Major section breaks 
        "```\n", //Code blocks
        "\n- ", "\n* ", "\n1. ", "\n| ", //Lists/tables
        "\n### ", "\n#### ", // Subsections
        "\n<", "\n</", // HTML components
        "\n---\n", "\n***\n", //Horizontal rules
        "\n\n", "\n", " " // Soft breaks
    ];

    const len = text.length;
    let chunkSize = 250;
    if (len > 2500){
        chunkSize = 250 + Math.floor((len - 2500) / 2500) * 200;
    }
    if (chunkSize > 1000){
        chunkSize = 1000;
    }

    //Based on Markdown Document Seperators
    const headerSplitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
        chunkSize: chunkSize, 
        chunkOverlap: 50,
        keepSeparator: true,
        separators: Seperators,
    });
    return await headerSplitter.createDocuments([text]);
}

async function main(){
    const files = getFilePaths('content/docs');

    for (const file of files){
        const text = await readFile(file, 'utf-8');
        const chunks = await headerChunking(text);

        for (const chunk of chunks){
            const embedding = await embeddings.embedQuery(chunk.pageContent);
            await storeInChroma(embedding, chunk.pageContent, path.basename(file));
        }

        console.log("STORED: " + path.basename(file) + " IN CHROMA DB");
    }

    // TEMP PRINTS FOR YAML
    console.log("--- Processing api-schema.yaml ---");
    const openApiDocs = await readFile('api-schema.yaml', 'utf-8');
    console.log("Read api-schema.yaml, length:", openApiDocs.length);
    const openAiDocsSummary = await sumarizeOpenAI(openApiDocs);
    console.log("YAML summary generated, length:", openAiDocsSummary?.length);
    console.log("YAML summary preview:", openAiDocsSummary?.slice(0, 300));
    const summaryChunks = await headerChunking(openAiDocsSummary);
    console.log(`YAML summary split into ${summaryChunks.length} chunks.`);

    for (const [i, chunk] of summaryChunks.entries()){
        const embedding = await embeddings.embedQuery(chunk.pageContent);
        await storeInChroma(embedding, chunk.pageContent, 'api-schema.yaml');
        console.log(`STORED: api-schema.yaml chunk ${i+1}/${summaryChunks.length} IN CHROMA DB`);
    }
}
main().catch((err) => {
    console.error("Script failed:", err);
  });
  