# kafka/producer.py
from kafka import KafkaProducer
import json

class KafkaRecipeProducer:
    def __init__(self, topic, bootstrap_servers="localhost:9092"):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        self.topic = topic

    def send_message(self, message):
        self.producer.send(self.topic, value=message)
        print(f"Message sent to topic {self.topic}: {message}")
