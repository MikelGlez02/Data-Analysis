# kafka/consumer.py
from kafka import KafkaConsumer
import os
import json

class KafkaRecipeConsumer:
    def __init__(self, topic, bootstrap_servers=os.getenv("KAFKA_BROKER", "localhost:9092")):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8"))
        )

    def consume_messages(self):
        for message in self.consumer:
            print(f"Message received from topic {message.topic}: {message.value}")
