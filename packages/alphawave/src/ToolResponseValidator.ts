import { PromptResponseValidator, Validation, ChatCompletionFunction, PromptResponse, ChatCompletionTool } from "./types";
import { PromptMemory, PromptFunctions, Tokenizer, Message } from "promptrix";
import { Schema } from "jsonschema";
import { JSONResponseValidator } from "./JSONResponseValidator";

/**
 * Validates tool calls returned by the model.
 * @remarks
 */
export class ToolResponseValidator implements PromptResponseValidator {
    private readonly _functions: Map<string, ChatCompletionFunction> = new Map();

    /**
     * Creates a new `ToolResponseValidator` instance.
     * @param tools Optional. List of tools supported for the prompt.
     */
    public constructor(tools?: ChatCompletionTool[]) {
        if (tools) {
            for (const tool of tools) {
                this._functions.set(tool.function.name, tool.function);
            }
        }
    }

    /**
     * Gets a list of the functions configured for the validator.
     */
    public get functions(): ChatCompletionFunction[] {
        const list: ChatCompletionFunction[] = [];
        this._functions.forEach(fn => list.push(fn));
        return list;
    }

    /**
     * Gets a list of the tools configured for the validator.
     */
    public get tools(): ChatCompletionTool[] {
        const list: ChatCompletionTool[] = [];
        this._functions.forEach((fn, name) => list.push({ type: 'function', function: fn }));
        return list;
    }

    /**
     * Adds a new function to the validator.
     * @param name Name of the function.
     * @param description Optional. Description of how the model should use the function.
     * @param parameters Optional. JSON Schema for functions parameters.
     * @returns The validator for chaining purposes.
     */
    public addFunction(name: string, parameters: Schema, description?: string): this {
        if (this._functions.has(name)) {
            throw new Error(`FunctionResponseValidator already has an function named "${name}".`);
        }

        this._functions.set(name, { name, description, parameters });
        return this;
    }

    /**
     * Validates the response.
     * @param memory Memory used to render the prompt.
     * @param functions Functions used to render the prompt.
     * @param tokenizer Tokenizer used to render the prompt.
     * @param response Response to validate.
     * @param remaining_attempts Number of remaining validation attempts.
     * @returns A `Validation` with the status and value.
     */
    public async validateResponse(memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: PromptResponse, remaining_attempts: number): Promise<Validation> {
        // Validate individual tool calls
        const tool_calls = response.message?.tool_calls ?? [];
        for (const tool_call of tool_calls) {
            // Ensure name is specified
            const function_call = tool_call.function;
            if (!function_call.name) {
                return {
                    type: 'Validation',
                    valid: false,
                    feedback: `Function name missing for tool call. Specify a valid function name.`
                };
            }

            // Ensure name valid
            if (!this._functions.has(function_call.name)) {
                return {
                    type: 'Validation',
                    valid: false,
                    feedback: `Unknown function named "${function_call.name}". Specify a valid function name for tool call.`
                };
            }

            // Validate arguments
            const functionDef = this._functions.get(function_call.name);
            if (functionDef) {
                const validator = new JSONResponseValidator(
                    functionDef.parameters,
                    `No arguments were sent with tool call. Call the "${function_call.name}" with required arguments as a valid JSON object.`,
                    `The function arguments had errors. Apply these fixes and call "${function_call.name}" function again:`
                );
                const args = function_call.arguments === '{}' ? null : function_call.arguments ?? '{}'
                const message: Message = { role: 'assistant', content: args };
                const result = await validator.validateResponse(memory, functions, tokenizer, { status: 'success', message }, remaining_attempts);
                if (!result.valid) {
                    return result;
                }
            }
        }

        return {
            type: 'Validation',
            valid: true
        };
    }
}
