from vector_store import build_vector_store

# Your Knowledge Base
docs = [
    #  RETURNS 
    "Clothing items can be returned within 7 days of delivery if they are unused, unwashed, and have original tags attached.",
    "Footwear can be returned within 10 days if it is unworn and in its original packaging.",
    "Innerwear, socks, and personal care products are non-returnable for hygiene reasons.",
    "Return pickup will be scheduled within 24 hours after a return request is approved.",
    "Refunds for prepaid orders are processed to the original payment method within 5–7 business days after the product reaches our warehouse.",
    "Cash on Delivery refunds are processed to the customer’s bank account or UPI within 7–10 business days.",

    #   EXCHANGE  
    "Size exchanges are allowed for clothing and footwear within 7 days of delivery.",
    "Exchange is subject to product availability.",
    "Only one exchange is allowed per order.",
    "Exchange is not available for products marked as Final Sale.",

    #   CANCELLATION  
    "Orders can be cancelled before they are shipped.",
    "Once an order is shipped, it cannot be cancelled and must be returned after delivery.",
    "Refunds for cancelled orders are processed within 24 hours.",

    #   SHIPPING  
    "Free delivery is available on orders above ₹999.",
    "A shipping fee of ₹99 is charged for orders below ₹999.",
    "Orders are typically delivered within 3–5 business days.",
    "Remote locations may take up to 7 business days for delivery.",

    #   WARRANTY  
    "Shoes come with a 30-day warranty against manufacturing defects.",
    "Electronics come with a 1-year manufacturer warranty.",
    "Warranty does not cover damage due to wear and tear, water, or misuse.",

    #   PAYMENTS  
    "We accept credit cards, debit cards, UPI, net banking, and wallets.",
    "Cash on Delivery is available for orders below ₹10,000.",
    "Failed payments are automatically refunded within 3–5 business days.",

    #   REFUNDS  
    "Refunds for returned products are processed after quality check at our warehouse.",
    "Shipping charges are non-refundable unless the return is due to a defective or incorrect product.",
    "Refund status can be tracked in the ‘My Orders’ section.",

    #   CUSTOMER SUPPORT  
    "Our customer support is available 24/7 via chat and email.",
    "Billing and payment issues are handled by a specialized support team.",
    "Escalated issues are resolved within 24–48 business hours.",
]

if __name__ == "__main__":
    print("Building Vector Database...")
    build_vector_store(docs)
