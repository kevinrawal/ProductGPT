from transformers import pipeline
from bs4 import BeautifulSoup


def get_context_from_html(html_content):
    """Extracts product information from HTML and formats it into a context string."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract product title
    title = (
        soup.find(id="productTitle").get_text(strip=True)
        if soup.find(id="productTitle")
        else "Title not available"
    )

    # Extract product price
    price = (
        soup.find("span", {"class": "a-price-whole"}).get_text(strip=True)
        if soup.find("span", {"class": "a-price-whole"})
        else "Price not available"
    )

    # Extract product rating
    rating = (
        soup.find("span", {"class": "a-icon-alt"}).get_text(strip=True)
        if soup.find("span", {"class": "a-icon-alt"})
        else "Rating not available"
    )

    # Extract number of reviews
    review_count = (
        soup.find(id="acrCustomerReviewText").get_text(strip=True)
        if soup.find(id="acrCustomerReviewText")
        else "Review count not available"
    )

    # Extract product specifications
    details = soup.find(id="productDetails_techSpec_section_1")
    details = (
        details.get_text(strip=True).replace("\n", " ").replace("\r", "")
        if details
        else "Details not available"
    )

    # Extract additional information from the feature bullets
    feature_bullets = soup.find(id="feature-bullets")
    if feature_bullets:
        features = "\n".join(
            [
                item.get_text(strip=True)
                for item in feature_bullets.find_all("span", {"class": "a-list-item"})
            ]
        )
    else:
        features = "Features not available"

    # Extract product description
    description = (
        soup.find(id="productDescription").get_text(strip=True)
        if soup.find(id="productDescription")
        else "Description not available"
    )

    # Format the information for LLM context
    context = f"""
    Product Title: {title}
    Price: {price}
    Rating: {rating}
    Review Count: {review_count}

    Product Details:
    {details}

    Features:
    {features}

    Product Description:
    {description}
    """

    return context


def get_answer_from_model(context, question):
    """Gets an answer from a pre-trained model based on the input context and question."""
    # Load the pre-trained model for question answering
    qa_pipeline = pipeline(
        "question-answering", model="distilbert-base-uncased-distilled-squad"
    )

    # Use the model to answer the question based on the context
    result = qa_pipeline(question=question, context=context)

    return result["answer"]
