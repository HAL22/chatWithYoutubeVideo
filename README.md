# chatWithYoutubeVideo

In this application we apply RAG principles to a Youtube video. We use an audio-to-text model (whisper-tiny) to transcribe a Youtube video. The text from the video is chunked and vectorized and timestamps are added as metadata.
When user ask a question about a video(or part of the video) a response will appear and also a clip from the video will be displayed.

Requirments:
- Install packages and libraries from requirements.txt and packages.txt
- OpenAI was used , so make sure you have a OpenAI API key configured
- Pinecone was used as a vector store , so make you have a Pinecone API key configured 

Future:
- "Chat" with any video, not just a Youtube video
- Deploy and use as a chrome extension