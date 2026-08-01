export class Filter {
    private type: string;
    private args: any;

    private constructor(type: string, args: any) {
        this.type = type;
        this.args = args;
    }

    public static eq(field: string, value: any): Filter {
        return new Filter("eq", { field, value });
    }

    public static neq(field: string, value: any): Filter {
        return new Filter("neq", { field, value });
    }

    public static gt(field: string, value: any): Filter {
        return new Filter("gt", { field, value });
    }

    public static gte(field: string, value: any): Filter {
        return new Filter("gte", { field, value });
    }

    public static lt(field: string, value: any): Filter {
        return new Filter("lt", { field, value });
    }

    public static lte(field: string, value: any): Filter {
        return new Filter("lte", { field, value });
    }

    public static in(field: string, value: any[]): Filter {
        return new Filter("in", { field, value });
    }

    public static contains(field: string, value: string): Filter {
        return new Filter("contains", { field, value });
    }

    public static and(...filters: Filter[]): Filter {
        return new Filter("and", filters.map(f => f.toJSON()));
    }

    public static or(...filters: Filter[]): Filter {
        return new Filter("or", filters.map(f => f.toJSON()));
    }

    public static not(filter: Filter): Filter {
        return new Filter("not", filter.toJSON());
    }

    public and(other: Filter): Filter {
        return Filter.and(this, other);
    }

    public or(other: Filter): Filter {
        return Filter.or(this, other);
    }

    public toJSON(): any {
        return {
            type: this.type,
            ...this.args
        };
    }
}
