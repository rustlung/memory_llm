"""Main bot module with Telegram handlers."""
import asyncio
import logging
import sys
from functools import wraps

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ChatAction

from src.config import Config
from src.memory_short import ShortMemory
from src.vectordb import VectorDB
from src.llm import LLMClient
from src.rag import index_company_data, needs_reindexing, retrieve_context


# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Global instances
short_memory: ShortMemory = None
vector_db: VectorDB = None
llm_client: LLMClient = None
bot: Bot = None


def async_action(action: ChatAction):
    """Decorator to send chat action while processing."""
    def decorator(func):
        @wraps(func)
        async def wrapper(message: Message, *args, **kwargs):
            # Start sending action
            action_task = asyncio.create_task(
                send_action_periodically(message.chat.id, action)
            )
            try:
                return await func(message, *args, **kwargs)
            finally:
                action_task.cancel()
                try:
                    await action_task
                except asyncio.CancelledError:
                    pass
        return wrapper
    return decorator


async def send_action_periodically(chat_id: int, action: ChatAction):
    """Send chat action periodically."""
    try:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action=action)
            await asyncio.sleep(4)  # Telegram suggests sending every 5 seconds
    except asyncio.CancelledError:
        pass


async def cmd_start(message: Message):
    """Handle /start command."""
    welcome_text = """👋 Привет! Я бот компании ТехноСервис.

Я могу ответить на вопросы о нашей компании, продуктах, тарифах и услугах.

Доступные команды:
/start - Показать это сообщение
/help - Справка
/reset - Очистить историю диалога

Просто задай мне вопрос!"""
    
    await message.answer(welcome_text)


async def cmd_help(message: Message):
    """Handle /help command."""
    help_text = """ℹ️ Справка по использованию бота

Я работаю в двух режимах:

1️⃣ **RAG-режим** (когда вопрос касается компании):
   - Использую базу знаний о компании
   - Отвечаю строго на основе фактов
   - Примеры вопросов: "Какие у вас тарифы?", "Как вернуть деньги?", "Ваши контакты?"

2️⃣ **Обычный диалог** (для общих вопросов):
   - Помню последние 10 сообщений
   - Могу вести диалог на любые темы

Команды:
/start - Приветствие
/help - Эта справка
/reset - Очистить историю диалога (забуду наш разговор)

Просто напиши мне свой вопрос! 💬"""
    
    await message.answer(help_text)


async def cmd_reset(message: Message):
    """Handle /reset command."""
    chat_id = message.chat.id
    short_memory.clear(chat_id)
    await message.answer("✅ История диалога очищена. Начнём сначала!")


@async_action(ChatAction.TYPING)
async def handle_message(message: Message):
    """Handle regular text messages."""
    chat_id = message.chat.id
    user_message = message.text
    
    logger.info(f"User {chat_id}: {user_message}")
    
    try:
        # Run blocking operations in executor
        loop = asyncio.get_event_loop()
        
        # Retrieve context from vector DB
        chunks, best_score = await loop.run_in_executor(
            None,
            retrieve_context,
            user_message,
            vector_db,
            llm_client,
            Config.TOP_K
        )
        
        logger.info(f"Best similarity score: {best_score:.4f}, threshold: {Config.RAG_THRESHOLD}")
        
        # Decide which mode to use
        if best_score >= Config.RAG_THRESHOLD:
            # RAG mode
            logger.info("Using RAG mode")
            response = await loop.run_in_executor(
                None,
                llm_client.rag_completion,
                user_message,
                chunks
            )
            mode = "RAG"
        else:
            # LLM with short memory mode
            logger.info("Using LLM_WITH_CONTEXT mode")
            history = short_memory.get_history(chat_id)
            
            system_prompt = """Ты — дружелюбный помощник. 
Отвечай вежливо и по существу. 
Если не знаешь ответа — скажи об этом честно."""
            
            response = await loop.run_in_executor(
                None,
                llm_client.chat_completion_with_context,
                user_message,
                history,
                system_prompt
            )
            mode = "LLM_WITH_CONTEXT"
        
        # Save to short memory
        short_memory.add_message(chat_id, "user", user_message)
        short_memory.add_message(chat_id, "assistant", response)
        
        logger.info(f"Response mode: {mode}, length: {len(response)}")
        
        # Send response
        await message.answer(response)
        
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        await message.answer(
            "😔 Извините, произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /reset для сброса диалога."
        )


async def on_startup():
    """Initialize bot on startup."""
    global short_memory, vector_db, llm_client, bot
    
    logger.info("Starting bot initialization...")
    
    # Validate config
    try:
        Config.validate()
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Initialize components
    short_memory = ShortMemory(max_size=Config.SHORT_MEMORY_SIZE)
    vector_db = VectorDB(Config.DB_PATH)
    llm_client = LLMClient(
        api_key=Config.OPENAI_API_KEY,
        chat_model=Config.OPENAI_CHAT_MODEL,
        embed_model=Config.OPENAI_EMBED_MODEL
    )
    
    logger.info("Components initialized")
    
    # Check if indexing is needed
    if needs_reindexing(Config.COMPANY_TXT_PATH, vector_db):
        logger.info("Indexing company data...")
        index_company_data(Config.COMPANY_TXT_PATH, vector_db, llm_client)
        logger.info("Indexing complete")
    
    logger.info("Bot is ready!")


async def main():
    """Main function to run the bot."""
    global bot
    
    # Create bot instance
    bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    
    # Register handlers
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_reset, Command("reset"))
    dp.message.register(handle_message, F.text)
    
    # Startup
    await on_startup()
    
    # Start polling
    logger.info("Starting polling...")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
