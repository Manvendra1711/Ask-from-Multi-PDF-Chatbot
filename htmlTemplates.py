css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .label {
  width: 15%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 14px;
}
.chat-message.user .label {
  color: #4CAF50;
}
.chat-message.bot .label {
  color: #2196F3;
}
.chat-message .message {
  width: 85%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="label">
        {{LABEL}}
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="label">
        {{LABEL}}
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''