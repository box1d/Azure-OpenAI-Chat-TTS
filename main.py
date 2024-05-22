import azure.cognitiveservices.speech as speechsdk
from openai import OpenAI
import asyncio
from collections import deque

# Replace with your subscription key and service region
subscription_key = "YourAzureSubscriptionKey"
service_region = "YourServiceRegion"

# Replace with your OpenAI API key
openai_api_key = "YourOpenAIAPIKey"

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

# Global event loop and task management
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
recognition_task = None
synthesizer_task = None  # Initialize synthesizer_task variable
# Store the last three conversations (each conversation includes user and assistant)
conversation_history = deque(maxlen=6)


async def speak_text_async(synthesizer, text):
    result_future = synthesizer.speak_text_async(text)
    tts_result = await loop.run_in_executor(None, result_future.get)
    if tts_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("\nSpeech synthesized successfully.")
    elif tts_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = tts_result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")


async def handle_streaming_response(synthesizer, text):
    global synthesizer_task  # Ensure global variable reference within function
    try:
        conversation_history.append({"role": "user", "content": text})
        messages = [
            {"role": "system", "content": "You are a chatbot, please answer all my questions concisely."}
        ] + list(conversation_history)

        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )

        response_text = ""
        buffer_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                buffer_text += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content,
                      end="")  # Print stream content

                # Check if it contains punctuation or sufficient length
                if len(buffer_text) > 30 and any(punctuation in buffer_text for punctuation in ['，', '。', '！', '？']):
                    segment, buffer_text = split_response(buffer_text)
                    await synthesize_text_segment(synthesizer, segment)

        # Handle remaining response text
        if buffer_text:
            await synthesize_text_segment(synthesizer, buffer_text)

        # Add assistant response to conversation history
        conversation_history.append(
            {"role": "assistant", "content": response_text})

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")


async def synthesize_text_segment(synthesizer, segment):
    global synthesizer_task  # Ensure global variable reference within function
    # Cancel the current synthesis task
    if synthesizer_task is not None:
        synthesizer_task.cancel()
        try:
            await synthesizer_task
        except asyncio.CancelledError:
            pass
    synthesizer.stop_speaking_async()  # Stop current synthesis
    # Use TTS to play the response text segment
    synthesizer_task = asyncio.create_task(
        speak_text_async(synthesizer, segment))
    await synthesizer_task


def split_response(response_text):
    # Only split when text length exceeds 30 characters
    if len(response_text) > 30:
        # Prefer splitting at punctuation
        for i in range(len(response_text) - 1, -1, -1):
            if response_text[i] in ['，', '。', '！', '？']:
                return response_text[:i+1], response_text[i+1:]

        # If no punctuation, try splitting at spaces or word boundaries
        for i in range(20, len(response_text)):
            if response_text[i] == ' ':
                return response_text[:i], response_text[i+1:]

    # If no suitable space or punctuation, split directly
    return response_text, ""


async def recognize_and_synthesize():
    # Ensure global variable reference within function
    global recognition_task, synthesizer_task
    speech_config = speechsdk.SpeechConfig(
        subscription=subscription_key, region=service_region, speech_recognition_language="zh-CN")
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    async def recognized(evt):
        # Ensure global variable reference within function
        global recognition_task, synthesizer_task
        text = evt.result.text
        print(f"RECOGNIZED: Text={text}")
        if len(text) > 0:
            if recognition_task is not None:
                recognition_task.cancel()
            recognition_task = asyncio.create_task(
                handle_streaming_response(synthesizer, text))

    def recognized_handler(evt):
        asyncio.run_coroutine_threadsafe(recognized(evt), loop)

    def stop_cb(evt):
        print('CLOSING on {}'.format(evt))
        recognizer.stop_continuous_recognition()

    recognizer.recognized.connect(recognized_handler)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    recognizer.start_continuous_recognition()

    print("Continuous recognition started. Press Enter to stop...")
    await loop.run_in_executor(None, input)
    recognizer.stop_continuous_recognition()

if __name__ == "__main__":
    loop.run_until_complete(recognize_and_synthesize())
