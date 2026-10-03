from flask import Flask, render_template, request, jsonify, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

from dotenv import load_dotenv
load_dotenv()

from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["endpoint"])
CHAT_ERRORS = Counter("chat_errors_total", "Failed chat requests")

app = Flask(__name__)

# Load the existing AstraDB collection once at startup and build the chain
vector_store = DataIngestor().ingest(load_existing=True)
rag_chain = RAGChainBuilder(vector_store).build_chain()


@app.route("/")
def index():
    REQUEST_COUNT.labels(endpoint="/").inc()
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def get_response():
    REQUEST_COUNT.labels(endpoint="/get").inc()
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    session_id = data.get("session_id") or "default"

    if not question:
        return jsonify(error="Please type a question."), 400

    try:
        result = rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return jsonify(answer=result["answer"])
    except Exception:
        CHAT_ERRORS.inc()
        app.logger.exception("RAG chain failed")
        return jsonify(error="Couldn't get an answer right now. Try again."), 500


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)