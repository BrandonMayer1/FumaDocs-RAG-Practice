import { ChromaClient } from 'chromadb';
import { OllamaEmbeddings } from '@langchain/ollama';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { readFile } from 'fs/promises'; 
import fs from 'fs';
import path from 'path';

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
    
    });//Potentially Change these parameters
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

}
main().catch((err) => {
    console.error("Script failed:", err);
  });
  